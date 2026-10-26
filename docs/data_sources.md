# 数据来源与许可核对

尚无数据被当作正式课程结果。数据下载不进入 Git；split manifests 与内容hash进入 Git。

## CIFAR-10

官方来源：[Alex Krizhevsky 数据页面](https://www.cs.toronto.edu/~kriz/cifar.html)，下载 CIFAR-10 Python archive，页面列 MD5 `c58f30108f718f92721af3b95e74349a`。页面确认 50000 train/10000 test、每类train5000；要求引用技术报告 Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images, 2009。

核对日期 2026-10-26。官方页面没有明确列出单独的数据集许可证，不能将 torchvision 源码许可证当成数据许可证。课程 PDF 明确指定该公开数据用于实验；仅在本地下载用于课程分析，不向仓库分发原始图像。**独立许可条款仍未确认**，公开再分发前须补证；不声称已完成完整许可审查。

划分：seed1729，每类val500/cal200，剩余4300；官方test独立命名空间。split为自建且显式标记；缺失课程协议时不能声称与教师文件逐索引一致。若教师提供标准split，保留本次记录，新增协议版本并重新验收，禁止偷偷覆盖已有split。

## 后续数据

FashionMNIST、Planetoid、SVHN及RAVDESS尚未下载，许可证/原始来源/具体版本待使用前核验。RAVDESS教师预处理包、Planetoid课程masks及颜色控制集缺失，详见requirements/missing_resources.md。

合成序列与2D目标仅按PDF实现；它们是明确规定的实验数据，不能与用于单元测试的`test_fixture`混淆。

## 软件参考

安装组合依据 [PyTorch 历史版本官方说明](https://pytorch.org/get-started/previous-versions/) 中2.7.1/cu126/torchvision0.22.1配对；不是宣称使用最新版本。确定性设置依据 [PyTorch 2.7 reproducibility](https://docs.pytorch.org/docs/2.7/notes/randomness.html)。
