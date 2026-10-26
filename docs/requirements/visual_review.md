# PDF 图像核对记录

2026-10-26：所有152页提取全文并阅读；以下20页已渲染为并排图像并实际查看。图像为本地审计缓存，原PDF不改。未声称152页都已逐页图像审校。公式实现前仍应针对对应公式复查原PDF。

| Lab | PDF页 | 核对内容 |
|---|---|---|
| 0 | 8、17 | SHA256 slices/取模/选项顺序；E1及E3阈值表、C checksum冲突；2std最小效应 |
| 1 | 5、9 | rho_T、gl、GI、AVD；scaled残差上限确为1.2e^c而非e^c/1.2；E1 chi |
| 2 | 6、8 | 模型尺寸/CPU变体；6种corruption×3级参数 |
| 3 | 3、6 | P2插值、P1/P3规则；序列长度/目标构造/三profile字段 |
| 4 | 4、12 | collapse_scale的trace阈值；rank undefined及trace恒等；必测细则 |
| 5 | 5、12 | GCN传播极限需按sqrt(degree+1)归一；测试阈值与报告map |
| 6 | 3、7 | Langevin噪声关系/离散方差；尾界；目标坐标与固定seed/grid |
| 7 | 6、13 | DDIM方差式、NFE；rare-mode参考频率含外部类；必测tau条文 |
| 8 | 6、11 | cal专用于conformal；ECE合成测试样本量确为10^5；OOD拒算规则 |
| 9 | 4、7 | 半开时间窗及gate约定；两encoder字段同slice不同mod确实同时存在 |

图像文件命名`sources/labN-visual-check.png`，均为本次实际渲染。提取用pypdf，渲染用pypdfium2；最初Poppler命令也成功渲染Lab0第8页，但pdftotext不可用，已采用bundled pypdf提取。
