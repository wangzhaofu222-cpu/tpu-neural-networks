# Controller-only instrumentation before formal feature/probe operations

Formal encoder training uses the committed6ff631c source/config throughout. The source-hash scope(src/tests/scripts/configs) remains clean; no training, loss, data, model or optimizer code is changed during encoder training.

Before any formal feature/probe operation starts, the outer controller adds a monotonic operationtimer, UTCstart, sourcehash and acontroller snapshot to those operations. Probe timers include the3candidate trainingruns andtest/kNN, so these times mustnot be added again to individualcandidate trainingtimes. Pythonprocessstartup is excluded explicitly. Encoderworker durations are already measured separately by the parent controller.

A controller subclass delegates every training step to the same ExperimentRunner and checks completion status. If the user requests a pause, an interruptedprobe is preserved as acheckpoint and never presented as acompletedtest result. This guard doesnot change computation when no pause is requested. The earlier budget operations retain theiroriginaltiming fields; no retroactive guessedtimings are inserted.

On retry the probe controller reuses only completed candidates with the exact same resolved-config hash andsourcehash, or resumes a matching interruptedcandidate through the existing full-state checkpoint mechanism. It never substitutes a different encoder, subset,regularization,seed orbudget. A fresh configuration still executes the full common training loop.
