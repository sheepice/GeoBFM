"""Opt-in finite-value instrumentation for bounded native-PPO fault replays."""
import json
from pathlib import Path
import types

import torch


def tensor_stats(value):
    value=value.detach()
    finite=torch.isfinite(value)
    safe=torch.where(finite,value,0.).float()
    return dict(shape=list(value.shape),dtype=str(value.dtype),nonfinite=int((~finite).sum()),
        absmax=float(safe.abs().max()),mean=float(safe.mean()))


def install_probe(algorithm, output, rank, starting_update):
    """Observe actual rollout/minibatch boundaries without changing the PPO loss."""
    path=Path(output);state=dict(update=starting_update,minibatch=0,current_batch=None,captured=False)
    logfile=path/f'numeric_probe_rank_{rank}.jsonl'

    def record(event, **details):
        with logfile.open('a') as file:
            file.write(json.dumps(dict(event=event,update=state['update'],
                minibatch=state['minibatch'],**details))+'\n')

    # Resume must use the saved adaptive LR, rather than the config's initial LR.
    saved_lr = algorithm.optimizer.param_groups[0]['lr']
    record('resume_lr',configured=algorithm.learning_rate,saved=saved_lr)
    algorithm.learning_rate = saved_lr
    # Object collectives deserialize every rank onto rank zero's CUDA device.
    # Tensor broadcast has identical values and no temporary GPU-zero model copy.
    def broadcast(self):
        with torch.no_grad():
            for model in (self._raw_actor,self._raw_critic):
                for value in model.state_dict().values():
                    if value.numel():
                        payload=value.contiguous()
                        torch.distributed.broadcast(payload,src=0)
                        if not value.is_contiguous():value.copy_(payload)
    algorithm.broadcast_parameters=types.MethodType(broadcast,algorithm)

    def capture(reason):
        if state['captured']:return
        state['captured']=True
        batch=state['current_batch']
        payload=dict(reason=reason,update=state['update'],minibatch=state['minibatch'],
            actor=algorithm._raw_actor.state_dict(),critic=algorithm._raw_critic.state_dict(),
            optimizer=algorithm.optimizer.state_dict(),learning_rate=algorithm.learning_rate,
            torch_rng=torch.random.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        if batch is not None:
            payload['batch']={k:getattr(batch,k) for k in
                ('observations','actions','values','advantages','returns','old_actions_log_prob','old_distribution_params')}
        target=path/f'numeric_fault_rank_{rank}.pt'
        torch.save(payload,target)
        record('captured',reason=reason,path=str(target))

    previous_generator=algorithm.storage.mini_batch_generator
    def batches(*args, **kwargs):
        for batch in previous_generator(*args, **kwargs):
            state['current_batch']=batch;state['minibatch']+=1
            values={key:tensor_stats(getattr(batch,key)) for key in ('values','returns','advantages')}
            record('minibatch_inputs',statistics=values)
            if any(v['nonfinite'] for v in values.values()):capture('nonfinite_minibatch_target')
            yield batch
    algorithm.storage.mini_batch_generator=batches

    previous_backward=algorithm._extra_backward
    def backward(self):
        previous_backward()
        stats={}
        for name,model in [('actor',self._raw_actor),('critic',self._raw_critic)]:
            gradients=[p.grad.detach().float() for p in model.parameters() if p.grad is not None]
            stats[name]=dict(nonfinite=sum(int((~torch.isfinite(g)).sum()) for g in gradients),
                absmax=max(float(torch.nan_to_num(g).abs().max()) for g in gradients),
                norm=float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))))
        record('before_gradient_clip',statistics=stats,learning_rate=self.learning_rate)
        if any(v['nonfinite'] for v in stats.values()):
            capture('nonfinite_gradient_before_clip')
            raise RuntimeError('Numeric probe captured nonfinite PPO gradient before optimizer step')
    algorithm._extra_backward=types.MethodType(backward,algorithm)

    def critic_hook(module, inputs, result):
        stats=tensor_stats(result)
        if stats['nonfinite'] or stats['absmax']>1000:
            record('critic_outlier',statistics=stats,
                observations={str(k):tensor_stats(v) for k,v in inputs[0].items(True,True) if torch.is_tensor(v)})
        if stats['nonfinite']:
            capture('nonfinite_critic_output')
            raise RuntimeError('Numeric probe captured nonfinite critic output')
    algorithm._raw_critic.register_forward_hook(critic_hook)

    previous_update=algorithm.update
    def update():
        state['minibatch']=0
        storage=algorithm.storage
        record('rollout',statistics={k:tensor_stats(getattr(storage,k)) for k in
            ('rewards','values','returns','advantages')},learning_rate=algorithm.learning_rate)
        try:return previous_update()
        except BaseException:
            capture('ppo_update_exception');raise
        finally:
            state['current_batch']=None;state['update']+=1
    algorithm.update=update
    record('installed',note='Diagnostic replay only; native PPO gradients/loss unchanged')
