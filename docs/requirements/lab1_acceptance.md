# Lab1 最终验收

工作完成；全部数值阈值未通过（1项）：缩放残差L96上界超出约2.15%，保留FAIL并完成逐块矩诊断。数值失败作为真实负面结果保留，不等于完整三轮成功。

| 验收项 | 状态 |
|---|---|
| E1_12_runs | PASS |
| E1_chi_within_20_percent | PASS |
| E1_after_epoch_all_depths | PASS |
| E2_12_runs | PASS |
| E2_growth_thresholds | FAIL |
| E2_growth_failure_diagnosed | PASS |
| E3_six_completed_runs | PASS |
| formal_clean_git | PASS |
| all_formal_real_data | PASS |
| CF5_observed_pair | PASS |
| six_report_sections | PASS |
| preregistered_before_all_E3 | PASS |
| E3_exactly_one_scientific_factor | PASS |
| full_epoch_step_counts | PASS |
| fixed_probe_and_gradient_sampling | PASS |
| GI_AVD_independent_recomputation | PASS |
| rho_independent_CPU_checkpoint_evaluation | PASS |
| formal_source_snapshots_match_Git | PASS |
| Lab0_all_preserved_artifact_bytes | PASS |
| all_automated_tests_pass | PASS |
| LaTeX_compiled | PASS |
| all_PDF_pages_visually_verified | PASS |
| PDF_hash_matches_QA | PASS |
| hypothesis_committed_content_and_original_working_bytes | PASS |
| preregistration_committed_content_and_original_working_bytes | PASS |

全部数值、RUN_ID、统计和限制见 `reports/lab1/results.json` 与六节俄语报告；5类必测和补充回归见 `docs/environment/tests-lab1-final.xml`。附加§14实验未标作已完成。
