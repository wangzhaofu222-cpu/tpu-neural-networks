# E1 主比较事前假设 / Предварительная гипотеза

Validation-selected F=gated, U=audio_plus_100105. Direction Δ=macroF1(F)−macroF1(U). Predict Δ>0: complementary speech/face cues may improve over parameter-matched unimodal capacity. This is a hypothesis, not an established result.

Exploration seed1 has already been observed. All main seeds1,2,3 will be retrained after this Git registration; this does not make exploration seed1 unseen. Shared frozen encoders, actor split, train-only normalization, input features SHA256 dcc28e968f9e846fd3c3fc64bbbf3caf35ec16ceb31d0d16968e212d029c7388, batch64, AdamW lr.001/wd.0001, 200epochs per trained head, validation-best macroF1, no head early stop, fixed evaluation masks/labels. Only F versus its capacity-matched unimodal U changes. Late fusion, if selected, requires two unimodal constituents each200epochs; its extra training cost is disclosed, not claimed compute-matched.

Falsification: Δ mean≤0 rejects predicted sign; mixed signs or |mean|<2 sampleSD means различие не установлено even if mean positive. Reverse established effects remain prediction FAIL. Metric noise is not removed through seed filtering. No test selection or seed replacement.

Selection evidence: E1_validation_selection.json. Main results must refer to the Git commit containing this unchanged file.

Concrete locked capacity: F has visual512/audio128, bias-free128-dimensional projections, scalar gate, hidden128, total100105 trained parameters. U is audio-only hidden731, total100155 parameters (relative mismatch0.0004994756). The visual+ alternative hidden192 has100040 parameters but was not selected as U because its validation macro-F1 was lower. F and U use the same frozen source features and paired batch-order seeds; differences do not establish a compute-matched comparison with a newly trained encoder. The test values from exploratory seed1 have been observed and are archived, not described as prospective independent evidence.
