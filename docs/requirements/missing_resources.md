# 缺失资源、歧义与阻塞

用户已于 2026-10-26 确认没有《Протокол данных курса》。学号已收到，学号本身不写文件。

| ID | 项目 | 影响 | 状态/处理 |
|---|---|---|---|
| RES-01 | 完整课程数据协议 | Lab 4 CF2 颜色控制集 §4.6；Lab 5 masks；Lab 8 近 OOD；Lab 9 actor/clip 协议 | BLOCKED；请向教师取得；不宣称公共替代等价 |
| RES-02 | CIFAR-10 / FashionMNIST 原始数据 | Lab 0/1/2/4/8 | CIFAR已从官方来源取得并通过官方MD5与本地SHA256校验；FashionMNIST尚未下载。Lab 0 按 PDF 明确 43k/5k/2k/10k 继续，split 方法显式记录；许可表述见docs/data_sources.md |
| RES-03 | 课程 Planetoid split 文件 | Lab 5 §6/8.1 | BLOCKED；可开发单元测试，正式验收须比对课程 masks |
| RES-04 | 课程 RAVDESS 224×224/10fps 人脸帧、16kHz mono、clip_id、face_found、actor roles、av_offset_ms | Lab 9；Lab 2/4/5 人脸附加项 | BLOCKED；公共原始视频不是教师预处理包 |
| RES-05 | Lab 2/4 真实已训练 checkpoint | Lab 8/9 | NOT RUN；必须先完成上游实验，不下载无关权重替代 |
| RES-06 | SVHN test、CIFAR-100 test 排除的 11 类名单 | Lab 8 远 OOD 必做；近 OOD 附加 | 远 OOD 待下载；11 类名单 BLOCKED，不自行选择 |
| RES-07 | CPU 温度遥测 | 本地大训练资源/热评估 | 未实测；用户随后要求忽略此等待项；保留预算与运行时限，不宣称温度安全 |
| RES-08 | Git 用户身份配置 | 真实提交及 hypothesis 前置记录 | 已按用户提供身份配置当前仓库；不再阻塞 |

## 教师文本冲突登记

**AMB-09-ENC（Lab 9 §7，p7，已图像核对）**：同一 `h9[0:2]` 分别 `%3` 选择 `lab9_visual_encoder=[cnn_lab2,vit_lab2,ssl_lab4]`，`%2` 选择 `lab9_encoder=[resnet18,vit_small]`。本profile实际是`cnn_lab2`与`vit_small`，确实冲突。用户随后明确授权自行决定：项目以更明确的上游来源`lab9_visual_encoder`为准（本例CNN Lab2），保留另一原始字段，标记`user_authorized_visual_encoder_precedence`。这是项目决定而非教师确认。Lab 2 的 canonical ViT 是 D384/L6，`vit_small`不足以唯一指代checkpoint；不得用其覆盖来源。课程数据包仍缺失，Lab9正式实验仍阻塞。

**AMB-00-E1（Lab 0 §22.2，p17）**：表格说模式 C first-batch checksum “不设阈值”，紧随文字又说 split/checksum 所有模式必须精确相同。保留两条并采用较严格的 checksum 一致检查；报告明确冲突，不借表格省略测量。

**AMB-00-COUNT（Lab 0 §12.1，p9）**：称“至少 13 次完整训练，模式 A”，但清单中 CF2 的 num_workers=2 按 §9.2 属模式 C。13 次计数仍保留；主 E0–E3 保持 A，CF2 作为明确 worker 干预标 C，不伪称 A。CUDA 附加 E1/E3 B/C 至多再 8 次。

**AMB-03-LEN（Lab 3 §6/9.1）**：正文称测试到 1024，但 profile L0=256 时 8L0=2048；以明确外推网格尝试，OOM 则记录首个不可行长度，不预先删点。

**AMB-06-ENERGY（Lab 6 §6.1/10.1）**：`E=||x||²/18+tanh(net)` 的残差跨度最多 2。two_blobs 中心基础能量 4/18，仍可低于原点，不能自行去掉 tanh 或扩大残差来追求更尖密度；低质量须如实诊断。

**AMB-07-TAU（Lab 7 §5.5/测试5）**：公式含 τ0=0，测试称所有元素在 [1,T]。将正时间网络调用网格 τ1..τK 与终止端点 τ0 单独存储，可同时满足。

Lab 0 的 PDF profile 示例已明确仅演示格式，不能作为真实 SHA256 参考向量。课件有个别错引公式编号；实现依公式内容并记录章节。
