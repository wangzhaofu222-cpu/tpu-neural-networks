# Metadata correction without changing any experimental outcome

The intentional E1 T/10 exact-DDPM diagnostic copied the baseline schedule dictionary before changingT. Schedule construction always recomputed beta/alpha/alpha_bar fromT andbeta arguments; the evaluation's own alpha_bar_T=.3635632480554922,protocol_valid=false,terminalKL andrawsamples were correct. However, an unused inherited scalar at diffusion.alpha_bar_T still held the fullT value. This is a metadata-only error,not an experiment or sampler error.

Original resolved YAML,passport,E1.json and preregisteredhashes remain unchanged. `schedule.resolved.json` explicitly preserves the oldscalar,flagsitsstaleness,andprovides the recomputed actual terminalvalue. The evaluator now refreshes this scalar for future schedule overrides. The independent audit replays actual samples fromT/beta andchecks actual protocol values. No training/evaluation threshold,randomseed,weight,sample or scientific conclusion was changed; no completed training was repeated.

Cosine beta_min/beta_max entries are unused lineararguments,not its effectivebeta limits. Numerical companions additionally record the actualcosinebeta range and.999cap. This clarifies the configuration without pretending that the cosine schedule is a linear schedule with thosebounds.
