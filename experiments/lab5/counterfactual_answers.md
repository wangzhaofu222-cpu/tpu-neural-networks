# Lab5 counterfactual answers — measured evidence

All conditions are transductive; PyG public masks were user-authorized. The teacher split remains independently UNVERIFIED. Test outcomes below are descriptive, never hyperparameter/depth selection input.

## Q1 — degree-preserving null versus MLP

Seed1: MLP 61.00% (`d16ae08008f6b91f6683-f4d55238a7904264a8e389e6ede11e4f`), null GCN 32.50% (`f1612dc13ae579d47fdc-1f33df1ac3aa47ec84dbd76b938a4bbc`), null GAT 26.90% (`d4019e4d9ffc0f51ebad-da107eb6b1fc474bad480d4ee24e01f3`). Three-seed MLP and null GCN values are in E2_pairs.csv. In (5.4), self-feature coefficient is 1/(di+1), while neighbor j contributes 1/sqrt((di+1)(dj+1)). Random neighbors can dilute class-relevant own features. For the measured degrees 1,3,168 these self weights are .5,.25,1/169 respectively. GAT (5.6) learns normalized attention including the self loop; it may reduce neighbor interference but has no guarantee of entirely rejecting random edges. The measured GAT−nullGCN difference is -5.60pp, not evidence that a random graph contains useful class edges. Null preserves degree information; model/hyperparameter differences also prevent treating MLP−GCN as a pure structural intervention.

## Q2 — predicted addition versus deletion

Before generating the addition graph, CF2_prediction.json recorded hE=0.809966, h0=0.179568, E=5278, added=2639, predicted new hE=0.599833 by weighted mixture. The actual hE is 0.599217, error -0.000616; finite nonedge sampling need not equal the class-frequency approximation. Added-graph GCN test=74.90% (`804407ffb262222bf788-5188bcee62fe409d81452e808519f81b`). Retrained deletion p=.5 test=77.10% (`83bdc4e3aa823dd29101-0d25d7abb9e542859b180983675318bb`). Addition minus deletion is -2.20pp: addition is more damaging in this fixed-seed comparison. The graph perturbations are different interventions, not an uncertainty estimate.

## Q3 — two deletion modes at p=.5

Original GCN=82.90%. ModeA=77.10%, drop=5.80pp (`03e1619b9899f2883b3d-e8688222db92423ab88987a7a88c7756`); modeB=77.10%, drop=5.80pp (`83bdc4e3aa823dd29101-0d25d7abb9e542859b180983675318bb`). 177/1000 test nodes become isolated. Small loss would not establish that the model ignores the graph: surviving neighbors remain homophilic and node features/self loops remain available. ModeA tests distribution shift; modeB permits parameter adaptation on that same damaged graph.

## Q4 — high cosine alone is insufficient

实测 plain_2 第 1 层：S=0.875197，Ê=0.0619819，probe=78.20%，该模型最佳层 probe=79.60%。分析 RUN_ID `49dc168afb79b6fc9815-380db0ebe0084bc0957d925932b8d12a`，probe RUN_ID `8153d5dd8492676db92b-0700efde84e442f4b5d6b4a47b385f35`。 ReLU gives a shared positive direction and cosine is insensitive to vector magnitude and small discriminative directions. Energy measures degree-corrected neighbor variation relative to total signal; the train→val probe measures retained label-relevant information. High S without lost probe accuracy does not establish harmful oversmoothing. Probe uncertainty and this one fixed split still limit the conclusion.

## Q5 — posthoc forbidden selection comparison

The immutable all_training_selection_lock.json selected depth 2 using val before E3 depth-test inspection. Posthoc test argmax is depth 2; they agree. The per-depth evaluation RUN_IDs are in E3_depths.csv. This posthoc computation changes no configuration, checkpoint or result. E1 had already exposed the separately required depth2 baseline test outcome; no claim of a wholly unseen depth2 test is made. Transduction permits X and edge access, not test-label selection. Selecting depth on test would reuse the evaluation labels and bias the final estimate even if the two argmax depths happen to agree.

For Q2, random added cross-class neighbors reduce edge homophily and can inject misleading messages. Random deletion largely preserves hE but reduces the available neighbors; the interventions damage different information. The measured one-seed difference does not quantify sampling uncertainty.
