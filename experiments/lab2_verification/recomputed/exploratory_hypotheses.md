# Exploratory E1 and E2 expectations

Recorded after launching the first LR candidate, before any formal run has
completed and before any final formal validation/test accuracy was available.
The original conditional E3 preregistration remains unchanged in its original
Git commit; this document is a supplementary explicit exploratory prediction.

E1: expect CNN test accuracy to exceed ViT at the smallest fraction under
S10000/A1. Locality and spatial weight sharing, equations (2.1)–(2.2), reduce
the need to learn these biases from 2150 examples. Expect that increasing
the number of distinct examples reduces the disadvantage of learning global
attention and position dependence in equations (2.3)–(2.5). Predict a smaller
absolute CNN advantage at r1 than r.05; a crossover is possible, not required.
An opposite ordering or growing gap rejects this exploratory expectation.

E2: expect mean corrupted accuracy to be below clean accuracy for each
architecture and each profile corruption. Expect patch shuffle to damage
CNN more than a mild contrast change: it disrupts local neighborhoods on
which equation (2.1) depends. ViT can also lose accuracy because learned
positions in equation (2.3) prevent exact permutation invariance. No leader
reversal is assumed; claim one only with opposite clean/mean-corrupted
signs and both gaps at least .01. Compare measured effects rather than
assigning robustness from architecture names.
