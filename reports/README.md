# Сборка отчётов / 报告构建

`tools/`仅后处理已保存测量或执行明确标记的诊断；不修改原始训练日志。Lab0的运行注册表、验收JSON与报告数值均来自真实RUN目录。训练环境与PDF导出环境分别锁定，见项目根`requirements.lock.txt`和本目录`requirements.lock.txt`。

Lab0主实验系列：`experiments/lab0/6116b9614f294c4c918500d92471fc4e/ledger.json`。主E2预定五个seed；CF4额外seed单独登记，不改变预定统计。性能复测保留原E4样本，并单独创建RUN_ID。

在项目根使用训练环境运行：

```powershell
& .\.conda-env\python.exe reports/tools/summarize_lab0.py --suite experiments/lab0/6116b9614f294c4c918500d92471fc4e --out reports/lab0
& .\.conda-env\python.exe reports/tools/validate_lab0_evidence.py --suite experiments/lab0/6116b9614f294c4c918500d92471fc4e --report reports/lab0
& .\.conda-env\python.exe reports/tools/markdown_to_tex.py reports/lab0/report.ru.md --out reports/lab0/main.tex
```

报告源文件`main.tex`使用T2A、俄语babel、amsmath、longtable、xurl和graphicx，图像相对路径位于同目录`images/`。可在已配置TeX的环境编译；本机内置预览器启动错误1060，不能据此声称LaTeX编译通过。

独立PDF导出使用Python3.12.14、ReportLab4.4.9和嵌入的Windows Times New Roman/Courier字体，输入仍是同一份俄语Markdown。运行`tools/render_report_pdf.py ... --out reports/lab0/report.pdf`，随后以pypdfium2渲染检查。最终PDF不是LaTeX编译成功的证据。PDF检查与构建状态见`lab0/build_status.json`。

不要在正式配对实验仍运行时提交Git或修改核心代码。完整训练系列及补充CF4使用固定提交；后续发布修改必须另留验证证据，不能改写旧manifest以声称它们来自新提交。
