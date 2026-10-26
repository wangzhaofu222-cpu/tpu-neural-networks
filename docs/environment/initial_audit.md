# 初始环境审计

2026-10-26 本机实际检查：Windows 11 build26200；默认Anaconda Python3.13.5，没有torch；pet环境存在；另有桌面旧课程环境Python3.10.20（只读检查），虽有torch但不满足Python≥3.11，不复用。当前项目只有10 PDF，无数据、代码或Git仓库。

NVIDIA驱动617.42；nvidia-smi可见RTX3060 Laptop GPU，6144MiB，初始约1024MiB占用、34°C、GPU util0%。驱动显示的CUDA13.4不等于PyTorch编译运行时；以后者`torch.version.cuda`为准。

初始D盘空余约67.9GB。后续psutil检查内存总17,024,741,376bytes、可用约3.66GB，磁盘空余约66.6GB（会随安装变化）；不把用户硬件型号推断成未实测内存值。

PowerShell个人profile调用conda.exe有DLL/日志权限错误；后续使用不加载profile的会话与`python -m conda`。libmamba在官方/conda-forge都异常声称缺pip，切换classic成功创建`.conda-env`，Python3.11.17，既有环境未修改。下载缓存`.conda-pkgs`/`.cache`均在项目内并被Git忽略。

CIM OS查询被限制；Windows版本改用Python platform/psutil核验。CPU温度未测；用户后来明确要求忽略温度监控等待，因此不再作为执行阻塞。仍保留线程限制、时限及长训练前预算门。

Git初始无身份。用户后来提供姓名/邮箱并已用于当前仓库本地配置；未修改全局配置。Lab9歧义用户授权自行决定，采用`lab9_visual_encoder`来源字段，不伪称教师确认。
