# Lab6 五个反事实问题（实测分析，不是正式报告）

## Q1:固定K下eta趋零会怎样？

eta=0.005时DK=0.892554，eta=0.02时DK=1.72865。固定K=50时，漂移总尺度随eta、累计噪声尺度随sqrt(K*eta)减小，链趋向初始点；小步长并不保证短链样本接近目标。

实测值、计算与 RUN_ID：

```json
{
  "measurements": [
    {
      "eta": 0.005,
      "MMD2": 0.052330830053531896,
      "Cov": 0.0,
      "Dmode": 0.7476,
      "DK": 0.8925535082817078,
      "outside_final_fraction": 0.0,
      "pre_reflection_outside_proposal_fraction": 0.0,
      "stable": true,
      "center_to_other_mode_fraction": 0.0,
      "RUN_ID": "134cc1ada1520a7482e7-d460ecfd9b1642fc9e2cf5b3c6f6d7ba",
      "source_RUN_ID": "90803714ee7813d313f0-7741207547b7491dabeba08a2be0959d"
    },
    {
      "eta": 0.02,
      "MMD2": 0.03717453020791317,
      "Cov": 0.0,
      "Dmode": 0.7582,
      "DK": 1.728647232055664,
      "outside_final_fraction": 0.0,
      "pre_reflection_outside_proposal_fraction": 0.0,
      "stable": true,
      "center_to_other_mode_fraction": 0.007000000216066837,
      "RUN_ID": "9566292dba4841aae029-5961859ab47d43d5bd13594a7378baa4",
      "source_RUN_ID": "90803714ee7813d313f0-7741207547b7491dabeba08a2be0959d"
    }
  ]
}
```

## Q2:最大eta的理想二次链和学习能量首先在哪方面偏离？

二次无边界链理论方差=1.33333，实测=1.33431，误差=0.074%。学习能量最大步长的stable=True、Cov=0、MMD²=0.203648。在最小eta=0.005时，Cov已是0.0、MMD²已是0.0523308；四个eta均数值稳定。因此本组首先暴露的是生成质量而非数值发散；在这四点网格的分辨率下，Cov和MMD都已偏离，不能捏造谁在更小eta先失败的顺序。稳定性仅说明没有数值发散，不能说明覆盖或样本质量正确。二次关系只约束其自身模型的方差，不预测学习能量的Cov/MMD²。

实测值、计算与 RUN_ID：

```json
{
  "quadratic": {
    "eta": 0.5,
    "sigma": 1.0,
    "effective_temperature": 1.0,
    "chains": 8,
    "burn_in": 10000,
    "draws_per_chain": 200000,
    "seed": 8104,
    "variance_per_chain": [
      1.336238127540144,
      1.3324346672229186,
      1.3365172933734948,
      1.3286486792753658,
      1.3366955842122967,
      1.3299271920735745,
      1.3375600488066122,
      1.336486736252325
    ],
    "measured_variance": 1.3343135410945914,
    "theory_variance": 1.3333333333333333,
    "relative_error": 0.0007351558209436337,
    "acceptance": "PASS",
    "method": "scipy lfilter executes all finite AR1 steps; eight independent innovations; no stationary analytic sampling; double precision",
    "RUN_ID": "e582a8a8c18deb14213c-a89bcdd573234772991b59efd1b4f2a3",
    "optimizer_steps": 0
  },
  "learned": {
    "eta": 0.5,
    "MMD2": 0.2036482206353436,
    "Cov": 0.0,
    "Dmode": 0.932,
    "DK": 4.511741638183594,
    "outside_final_fraction": 0.0,
    "pre_reflection_outside_proposal_fraction": 0.003588,
    "stable": true,
    "center_to_other_mode_fraction": 0.029200000688433647,
    "RUN_ID": "181a984d49ebf969ffe9-a4f272342fd9498899f4988990687960",
    "source_RUN_ID": "90803714ee7813d313f0-7741207547b7491dabeba08a2be0959d"
  },
  "smallest_eta": {
    "eta": 0.005,
    "MMD2": 0.052330830053531896,
    "Cov": 0.0,
    "Dmode": 0.7476,
    "DK": 0.8925535082817078,
    "outside_final_fraction": 0.0,
    "pre_reflection_outside_proposal_fraction": 0.0,
    "stable": true,
    "center_to_other_mode_fraction": 0.0,
    "RUN_ID": "134cc1ada1520a7482e7-d460ecfd9b1642fc9e2cf5b3c6f6d7ba",
    "source_RUN_ID": "90803714ee7813d313f0-7741207547b7491dabeba08a2be0959d"
  },
  "metric_noise_MMD2": 3.5877382982718586e-05
}
```

## Q3:错误噪声匹配对应什么平稳规律？

eta=0.5，固定sigma=0.2，Teff=0.04；离散平稳方差=0.05346，理论sigma²/(2eta−eta²)=0.0533333。连续过程的理想二次分布方差是Teff，有限步长离散链多出1/(1−eta/2)因子，不能混为一谈。

实测值、计算与 RUN_ID：

```json
{
  "measurements": {
    "eta": 0.5,
    "sigma": 0.2,
    "effective_temperature": 0.04000000000000001,
    "chains": 8,
    "burn_in": 10000,
    "draws_per_chain": 200000,
    "seed": 8105,
    "variance_per_chain": [
      0.05360087114283015,
      0.05290674880752666,
      0.05351830082788905,
      0.053571349806320556,
      0.05362883108032693,
      0.053403290963670295,
      0.05365623647802948,
      0.05339473800927752
    ],
    "measured_variance": 0.05346004588948383,
    "theory_variance": 0.053333333333333344,
    "relative_error": 0.002375860427821646,
    "acceptance": "PASS",
    "method": "scipy lfilter executes all finite AR1 steps; eight independent innovations; no stationary analytic sampling; double precision",
    "RUN_ID": "4a189c7de430dae3185e-c1f0345d63354e0781a36957c9b4cef4",
    "optimizer_steps": 0
  }
}
```

## Q4:更好的likelihood是否意味着更好的生成样本？

取按深度排列最深的两个条件。深度8：valNLL=1.13468，MMD²=9.68068e-05；深度16：valNLL=1.13971，MMD²=7.43208e-05。这对实测中NLL与MMD排序不一致。MMD差只有2.2486e-05，噪声水平是3.58774e-05，不能把微小排序差夸大为可靠的生成优势。NLL评价数据点密度，MMD评价生成分布，有限样本有噪声；二者没有普遍的单调蕴含关系。

实测值、计算与 RUN_ID：

```json
{
  "measurements": [
    {
      "depth": 8,
      "val_NLL": 1.1346770509600639,
      "val_NLL_SEM": 0.014648499866835895,
      "MMD2": 9.680675360934643e-05,
      "Cov": 1.0,
      "Dmode": 0.010800000000000008,
      "parameters": 69648,
      "sample_seconds": 0.009394300000167277,
      "exact_test_NLL": 1.151692914879322,
      "RUN_ID": "a0314c5c3d42f5040040-6e051e63939e4c07846ff27ff10ff079",
      "source_RUN_ID": "c8fbbd1ec0df1fab5db6-c8d9aac7e54540b1b3319aa2d6670087",
      "invertibility": {
        "dtype": "float64 with converted learned weights",
        "test_points": 5000,
        "median_epsilon": 1.266065247384982e-16,
        "max_epsilon": 2.5293689273457983e-15,
        "autograd_points": 100,
        "max_logdet_error": 2.886579864025407e-15,
        "norm20_points": 100,
        "norm20_seed": 6201,
        "norm20_median_epsilon": 3.6885292664327447e-13,
        "norm20_max_epsilon": 2.1009554130252733e-10
      }
    },
    {
      "depth": 16,
      "val_NLL": 1.1397115485787392,
      "val_NLL_SEM": 0.013861994293237105,
      "MMD2": 7.432077492453004e-05,
      "Cov": 1.0,
      "Dmode": 0.004999999999999974,
      "parameters": 139296,
      "sample_seconds": 0.022970999999415653,
      "exact_test_NLL": 1.1583168642520905,
      "RUN_ID": "6f7020df7a993543ca96-e9094c92df4a4acdb304487a46618133",
      "source_RUN_ID": "196e303e9fb1bd3e788e-314f29270df449c486da4dd862908b03",
      "invertibility": {
        "dtype": "float64 with converted learned weights",
        "test_points": 5000,
        "median_epsilon": 1.5873190281692294e-16,
        "max_epsilon": 2.799137374830395e-14,
        "autograd_points": 100,
        "max_logdet_error": 4.3076653355456074e-14,
        "norm20_points": 100,
        "norm20_seed": 6201,
        "norm20_median_epsilon": 3.029575541881082e-07,
        "norm20_max_epsilon": 0.00010701972483449962
      }
    }
  ],
  "selection_rule": "two deepest E2 conditions by numeric depth; descriptive counterfactual, never changes validation selection",
  "metric_noise_MMD2": 3.5877382982718586e-05
}
```

## Q5:数据附近低能量是否证明模式间密度低？

不能。模式中点的三个logdensity及整条连线已实测；EBM项明确为Omega网格归一化估计，Flow项为精确密度。数据能量低没有单独约束中点能量、归一化常数或生成链覆盖。目标高斯混合的中点密度很小但非零，支持仍是整个平面；实测桥不能单独证明严格断连支持的拓扑限制。结合有限深度、基础分布及优化讨论桥强度，不预设桥必然明显，也不把所有差异解释成拓扑的单一因果作用。

实测值、计算与 RUN_ID：

```json
{
  "measurements": [
    {
      "seed": 1,
      "EBM_RUN_ID": "d20b01c7307c0257f2dd-15fc31e7e1964437b127450ffaac2fe0",
      "Flow_RUN_ID": "cbfee41a37a5abc05267-03ff7991347b4adcb139e792bac0563e",
      "true": -16.480539160917736,
      "EBM_grid_estimate": -4.887737274169922,
      "Flow_exact": -23.625484466552734
    },
    {
      "seed": 2,
      "EBM_RUN_ID": "5146ebd52e80abca688c-8b1c99ce0b90452a8a5e9c185a9f3181",
      "Flow_RUN_ID": "bab2188b75245e2f8cc6-eb1fc37eee9946628b3aa16785a17233",
      "true": -16.480539160917736,
      "EBM_grid_estimate": -4.910131931304932,
      "Flow_exact": -7.298760414123535
    },
    {
      "seed": 3,
      "EBM_RUN_ID": "a8da75f7a516997e4e82-8ccd9a201e074f289e6c29682a2d5a43",
      "Flow_RUN_ID": "2a9a084eb89b7f2d68d6-5b4d9d6c2aa5417aa7f56898bf06b98e",
      "true": -16.480539160917736,
      "EBM_grid_estimate": -4.826334476470947,
      "Flow_exact": -5.690201282501221
    }
  ]
}
```
