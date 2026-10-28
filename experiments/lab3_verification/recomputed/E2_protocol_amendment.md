# E2 protocol amendment before isolated measurements

Actual Windows PyTorch2.7.1 build does not include Flash Attention. Forced Flash fails with `No available kernel`; the failure and capability-probe warnings are preserved. Forced EFFICIENT_ATTENTION supports both forward and backward under deterministic mode and will be the fused implementation. No silent fallback. Both E2 backends remainfloat16.

The initial serial same-process profiling pilot showed retained allocator baseline across lengths. Those five explicit profiles remain as pilot records and are excluded from primary slope fitting. Each primary length/backend and each CF5 width now runs in its own fresh process, preventing cross-condition retained tensors or library-workspace accumulation. Both actual forward/training peaks remain allocator measurements, not hL² estimates.

This changes only profiling operations and backend dispatch for alternatives; explicitfloat32 training mathematics, data, loss, seeds, optimizer and allE1 results are unchanged. Original exploratory hypotheses are preserved verbatim. This amendment is committed before the revised E2 measurements and before any new E3 training.
