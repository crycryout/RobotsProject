# Robots 课程项目｜Cai Renjie

**2026 年 11 月 26 日 19:00（北京时间）**。查看实际运行、代码修复和已验证结果，可读[本次工作总结](本次工作总结.md)。上台可直接打开 [五页投影版 PDF](汇报.pdf) 或 [带逐页演讲备注的 PowerPoint](汇报.pptx)，旁边放 [约三分钟讲稿与问答](讲稿与问答.md)。

| 需要用到的材料 | 文件 |
|---|---|
| 说明个人职责、本次工作和团队已有实现 | [个人工作说明](个人工作说明.md) |
| 讲清模型如何在仿真中行动 | [环境与 rollout 流程图](figures/rollout_flow.png) · [成功与超时案例，包括原视频](cases/案例说明.md) |
| 解释比较是否公平 | [一页评测协议 PDF](评测协议.pdf) · [配对结果与区间 CSV](data/paired.csv) |
| 展示数字并回答逐任务问题 | [一页结果表 PDF](结果表.pdf) · [完整结果表及来源](结果表.md) · [汇报表 CSV](data/report_table.csv) · [累计环境步数及耗时图](figures/training_env_steps.png) · [按轮训练曲线](figures/training_curve.png) · [固定模型评测图](figures/independent_eval.png) |
| 核对实验依据 | [运行环境与复现命令](复现.md) · [四组原始回合快照](data/source_records.zip) · [逐回合索引](data/eval_episodes.csv) · [完整原始 Base 筛选归档](data/screening_base_20261123.zip) · [团队 v2 报告与原表的核对](evidence/v2报告核对.md) |

第 4 页可以这样说：“每个训练臂只用一个训练 seed。固定评测只用 default 指令，在已筛选的原 train 候选池中取 10 个未参与 v2 更新的任务。原始 Base、T′、A-video、A-action 在 120 个起点上分别成功 51、51、48、51 次。两个辅助方案相对 T′ 的配对区间都跨零。” 正式 test split 本轮未测。三组训练策略核对了 HDF5 物理初态，Base 的筛选原表只能核对协议键。[H800 共享原表与核查依据](evidence/共享路径核查.md)
