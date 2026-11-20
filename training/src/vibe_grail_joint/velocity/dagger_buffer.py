"""Bounded, source-balanced DAgger aggregation, with fresh/history sampling."""

import torch


class DaggerBuffer:
    """Per-source random-key reservoirs. Only frozen features may be cached.

    Fresh data stays separate until PPO finishes. Retaining the largest random
    keys gives a uniform reservoir across iterations, without unbounded storage.
    Each source/speed gets an equal quota and is sampled equally when present.
    """

    def __init__(self, capacity=8192, strata=9, seed=0, *, sampling_groups=None):
        if capacity < strata or strata < 1:
            raise ValueError("DAgger capacity must cover every source/speed stratum")
        self.capacity, self.strata = int(capacity), int(strata)
        self.quota = capacity // strata
        self.sampling_groups = tuple(range(strata) if sampling_groups is None else sampling_groups)
        if len(self.sampling_groups) != strata:
            raise ValueError('DAgger sampling groups must cover every stratum')
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.fresh, self.history = {}, {}
        self.seen = torch.zeros(strata, dtype=torch.long)

    def _merge(self, old, new):
        data = new if old is None else torch.cat([old, new], 0)
        if len(data) > self.quota:
            data = data[data["_priority"].topk(self.quota).indices]
        return data.clone()

    def add(self, obs, labels, strata):
        # Called during inference-mode rollout: replay tensors must be ordinary
        # tensors so trainable attention can save their inputs for backward.
        with torch.inference_mode(False):
            data = obs.detach().to("cpu").clone()
            data["_teacher_action"] = labels.detach().to("cpu").clone()
            data["_priority"] = torch.rand(len(data), generator=self.rng)
            strata = strata.to("cpu")
            if ((strata < 0) | (strata >= self.strata)).any():
                raise ValueError("Unknown DAgger reference stratum")
            for key in strata.unique().tolist():
                sample = data[strata == key]
                self.seen[key] += len(sample)
                self.fresh[key] = self._merge(self.fresh.get(key), sample)

    def _sample_pool(self, pool, count):
        groups = {}
        for key in sorted(pool):
            groups.setdefault(self.sampling_groups[key], []).append(key)
        members = list(groups.values())
        # A shuffled balanced assignment also covers small minority sources.
        assignment = torch.arange(count) % len(members)
        assignment = assignment[torch.randperm(count, generator=self.rng)]
        pieces = []
        for i, keys in enumerate(members):
            n = int((assignment == i).sum())
            if len(keys) > 1:
                offset = int(torch.randint(len(keys), (), generator=self.rng))
                keys = keys[offset:] + keys[:offset]
            for j, key in enumerate(keys):
                amount = n // len(keys) + int(j < n % len(keys))
                if amount:
                    pieces.append(pool[key][torch.randint(len(pool[key]), (amount,), generator=self.rng)])
        return torch.cat(pieces, 0)

    def sample(self, count, device):
        if not self.fresh:
            raise RuntimeError("DAgger update requires student-collected fresh samples")
        old_count = count // 2 if self.history else 0
        fresh = self._sample_pool(self.fresh, count - old_count)
        data = torch.cat([fresh, self._sample_pool(self.history, old_count)], 0) if old_count else fresh
        data = data.to(device).clone()
        labels = data.pop("_teacher_action")
        data.pop("_priority")
        return data, labels

    def commit(self):
        for key, value in self.fresh.items():
            self.history[key] = self._merge(self.history.get(key), value)
        self.fresh.clear()

    def state_dict(self):
        return dict(version=1, capacity=self.capacity, strata=self.strata,
                    history=self.history, fresh=self.fresh, seen=self.seen,
                    rng=self.rng.get_state(), sampling_groups=self.sampling_groups)

    def load_state_dict(self, state):
        if (state["version"], state["capacity"], state["strata"]) != (1, self.capacity, self.strata):
            raise ValueError("DAgger replay layout differs from saved run")
        if tuple(state.get('sampling_groups', range(self.strata))) != self.sampling_groups:
            raise ValueError('DAgger replay sampling groups differ from saved run')
        self.history, self.fresh, self.seen = state["history"], state["fresh"], state["seen"]
        self.rng.set_state(state["rng"])

    def __len__(self):
        return sum(map(len, self.history.values())) + sum(map(len, self.fresh.values()))
