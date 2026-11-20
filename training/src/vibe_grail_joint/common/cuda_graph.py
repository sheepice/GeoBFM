"""Machine-local fallback for CUDA runtimes that retain graph allocations."""


def build_sensor_env(env_type, cfg, *, disable_graph_capture=False, disable_warp_mempool=False, **kwargs):
    """Use mjlab's eager simulation when requested by the machine profile.

    Physics and task settings stay identical. The temporary process-local class
    override is restored even on failure; each worker builds one environment on
    its main thread. Installed dependencies are never edited.
    """
    if disable_warp_mempool:
        if not disable_graph_capture:
            raise ValueError('Disabling the Warp memory pool requires eager simulation')
        import warp as wp
        # Collision temporaries are allocated during every rollout, after construction.
        wp.set_mempool_enabled(kwargs['device'], False)
    if not disable_graph_capture:
        return env_type(cfg, **kwargs)
    from mjlab.sim.sim import Simulation
    original_check = Simulation._should_use_cuda_graph
    Simulation._should_use_cuda_graph = lambda sim: False
    try:
        return env_type(cfg, **kwargs)
    finally:
        Simulation._should_use_cuda_graph = original_check
