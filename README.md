# RobotsProject

机器人课程 Final Project：RL Post-Training for World-Action Models。

本仓库当前是蔡仁杰负责的**仿真环境、rollout 基础设施与评估**的实施计划入口。尚未交付运行代码或 H800 实验结果。

## 从这里开始

阅读 [蔡仁杰详细实施计划](docs/RENJIE_EXECUTION_PLAN.md)。它包含 P0–P6 阶段、接口规范、数据 schema、评估协议、资源预算、验收清单及完整 Codex 执行指令。

你可以先独立完成环境、策略无关采集器、轨迹校验、统一评估和交接规范；真实 WAM、辅助奖励及 RL 更新的最终验收需要团队接入。

## H800 上执行

1. 克隆本仓库；如果已克隆，先保存本地工作，再更新到当前 main。
2. 在仓库目录启动 Codex。
3. 将下面指令交给 Codex。

```text
阅读 README.md 和 docs/RENJIE_EXECUTION_PLAN.md，依照第 13 节执行指令实施 P0–P5。先做真实 H800 与渲染预检，再完成可独立交付的仿真环境、rollout collector、轨迹校验、评估和队友交接接口。按阶段验收并在 docs/STATUS.md 留下实际命令与证据；不要只写计划、不要将 mock 结果当成 WAM 实验、不要默认为自己接管队友的奖励设计与 RL 训练器。被真实模型或硬件依赖阻塞时，记录阻塞并继续可完成部分。
```

## 执行前必须知道

- **RoboLab 官方要求 RTX GPU。** 不把两张 H800 当作已满足 RoboLab 的仿真/渲染条件；先走 ManiSkill 实测主线。
- **state 与 RGB 分别验收。** 状态环境可运行不代表视觉 WAM 的链路已完成。
- **动作维度相近不代表语义兼容。** DROID 模型接入其他控制器必须确认坐标系、单位、归一化与控制频率。
- **目前未提供教师完整评分细则。** 本计划依据用户上传 Proposal 的分工，不代表整个团队课程项目已经完成。

官方技术来源与详细限制见计划第 1、14 节。原 Proposal PDF、模型权重和实验数据没有上传到本公开仓库。
