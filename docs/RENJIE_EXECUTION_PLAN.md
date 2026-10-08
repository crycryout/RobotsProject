# 蔡仁杰独立实施计划：仿真环境、Rollout 与评估

版本：2026-10-08。目标仓库：crycryout/RobotsProject。

## 0. 结论与当前状态

你可以在队友尚未交付 WAM 和 RL 训练器时，独立完成一套可运行、可复现、可交接的仿真采集与评估基础设施。不能独立宣称完成的是：真实 WAM 的适配正确性、预测一致性奖励有效性、RL 后训练效果和最终泛化结论。

本计划依据用户提供的一页 `RL WAM.pdf` Proposal，保留原有分工。它不是教师完整评分细则；目前未提供机器人课程最终 rubric、提交模板或准确截止日期，因而不能保证仅完成本计划就满足全部课程评分要求。

本次检查时，GitHub 仓库为空，默认分支 main。本次交付仅包含计划和执行入口文档；没有声称已经安装模拟器、运行 H800、完成代码或取得实验结果。以下目录、命令和测试都是待服务器 Codex 实现的目标。

### 0.1 原有分工与边界

| 工作 | Proposal 负责人 | 本计划的处理 |
|---|---|---|
| 仿真环境、rollout infrastructure、evaluation | Cai Renjie | 本计划主线 |
| WAM integration、state-prediction alignment、reward design | Wang Junchang | 提供适配接口与数据，不擅自代替其科学设计 |
| experiment management、ablation analysis | Wang Shaohang | 提供统一命令和结果格式，避免重复建设调度系统 |
| RL 优化算法、反向传播、checkpoint 更新 | Proposal 未明确单独归属 | 记为待团队确认；不默认为蔡仁杰或实验管理员负责 |

“Rollout infrastructure”指策略调用、环境执行、轨迹保存、边界处理和评估。它不自动包含为扩散式 WAM 实现 PPO/GRPO、计算可训练策略概率或设计 RL loss。

### 0.2 可独立交付的五个成果

1. H800 环境与渲染诊断、可复现安装说明。
2. ManiSkill 小任务集、统一环境和策略接口。
3. 支持动作块、向量环境和断点恢复的轨迹采集器。
4. 公平评估、轨迹验证、性能测量及可审计结果。
5. 队友接入规范、示例数据和证据化进度记录。

按下文 P0–P5/P6 的工作包计算，除真实模型验收外，多数模块可以现在完成。不要用未经测量的“已完成 80%”替代逐项证据。

## 1. 技术路线与必须提前知道的限制

### 1.1 主线：ManiSkill，先 state，再真实 RGB

先用 `PushCube-v1`、`PickCube-v1` 两个候选任务建立小闭环，第三个任务只在前两者通过后增加。任务 ID、robot、control_mode、camera key 必须从锁定版本实际注册表和源码确认。初始调试可使用该版本支持的 `pd_ee_delta_pose`，但它不是 DROID 动作的默认兼容映射。

采用 state 观测跑通模拟器和数据管线后，必须单独验收相机 RGB 观测。state-only 的成功不等于 WAM 所需视觉链路完成；没有真实相机帧，状态应为 `PARTIAL / RGB_BLOCKED`。

环境早期可使用随机、合法的静止/保持控制以及官方可用的演示回放。随机策略成功率为 0 并不等于框架失败；反之，回放成功也不证明 WAM 能控制机器人。

### 1.2 RoboLab：条件分支，不在纯 H800 上默认投入

查阅时官方 RoboLab README 要求 NVIDIA RTX GPU，且基于 Isaac Sim/Isaac Lab。两张 H800 的 CUDA 算力不能替代 RTX 渲染兼容性。`--headless` 是不显示窗口，不是去掉视觉渲染要求。[S4]

因此：

- P0 先记录服务器实际 GPU 和渲染能力；若仅有 H800，RoboLab 分支标记 `BLOCKED_HARDWARE`，不要耗费几天尝试绕开硬件要求。
- 如果团队另有符合要求的 RTX 机器，可考虑 RTX 负责仿真、H800 负责模型推理；必须实测网络往返与图像传输成本。
- 用户曾提及的其他 RTX 资源不能视为当前已获分配；不要自动连接或占用未指定机器。
- 独立阶段先完成 ManiSkill 和后端无关的数据规范。只有确认 RTX 可用且官方小例子跑通，才实现 RoboLab 后端。

### 1.3 官方模型路径：参考实现优先，但不假设训练可用

NVIDIA 已提供 Cosmos policy server 与 RoboLab client 的示例，包含 Edge-Policy-DROID 和 `BananaInBowlTask` 路径。[S5] 这证明有现成接入参考，不证明当前 H800 服务器可以运行模拟器，也不证明服务支持训练所需梯度、策略概率或预测帧导出。

Cosmos 模型卡声明 DROID 8D 动作；不能把 8D 输出截成 7D 后塞入 ManiSkill 控制器。需要队友提供动作语义、坐标系、单位、归一化及时间尺度。[S6]

V-JEPA 2-AC 是带动作条件的潜空间世界模型，涉及基于目标图像的规划；不能把它当成直接替换 Cosmos 的轻量同接口策略。是否替换属于团队方法决策。[S7]

### 1.4 版本策略

不要无条件覆盖服务器现有 PyTorch/CUDA。先记录实际版本，在独立 venv/conda 环境中安装；模型服务和模拟器允许分属不同环境，通过明确接口连接。完成一次成功安装后锁定 Python、torch、mani_skill、SAPIEN、Gymnasium 和其他依赖，记录上游 commit。容器方案记录镜像 digest，而非只有 `latest`。

官方主分支可能更新。本计划引用 2026-10-08 查阅的资料，不把搜索页面标注的版本当成必须安装的最新稳定版本。

## 2. 完成状态与时间预算

### 2.1 按证据验收，不按日历打勾

| 阶段 | 单个目标 | 是否依赖队友 | 主动开发时间估计 | 完成证据 |
|---|---|---|---|---|
| P0 | 确认硬件、依赖和渲染能力 | 否 | 2–6 小时 | preflight.json、环境锁、决策记录 |
| P1 | 跑通环境和策略接口 | 否 | 4–8 小时 | 两任务 smoke、观测样本 |
| P2 | 正确采集并恢复轨迹 | 否 | 6–12 小时 | 合法轨迹、边界测试、恢复报告 |
| P3 | 生成可复现评估结果 | 否 | 4–8 小时 | episode 表、summary、测试清单 |
| P4 | 建立资源预算与可用控制策略 | 否，官方资源须可用 | 3–6 小时 | throughput 表、非随机控制证据 |
| P5 | 交付模型/奖励/训练接口 | 规范不依赖；真实验收依赖 | 3–6 小时 | contract、fixture、接入检查 |
| P6 | 集成队友真实 WAM 和训练 checkpoint | 是 | 4–12 小时起 | 真模型闭环、prediction 对齐记录 |

P0–P5 主动开发合计约 22–46 小时，是规划估计，不含下载、环境排障等待和长实验。Codex 可能加快实现，但不会消除硬件兼容和科学不确定性。不要照搬 Proposal 的“约 13 小时训练”作为整个项目预算；缺少模型/任务/步数/精度/更新策略/测量日志时它只是待核实陈述。

### 2.2 默认资源边界

初次执行只占用一张空闲且允许使用的 H800。另一张保留，待单卡流程正确后可跑独立实验；不先引入 DDP、Ray、Slurm 或多机系统。不要停止其他人的进程。

建议独立阶段默认累计 GPU 预算 12 GPU-hours，单个验证任务最多 2 小时；这是本计划设置的成本上限，不是速度承诺。耗尽后保存结果、报告剩余工作，继续 CPU 能完成的任务。仅在实际需要时由用户调整预算。

GPU-hours = 各进程实际占用 GPU 数量 × wall-clock 小时的累计；等待和 CPU 工作另报。性能数据必须记录设备是否独占。

## 3. P0：环境与兼容性预检

### P0.1 记录服务器状态

实现 `python -m robots_project.cli.preflight --output outputs/preflight.json`。采集 OS、CPU/RAM、可用磁盘、Python、驱动、GPU 型号/UUID/显存、CUDA runtime、torch CUDA 可用性和依赖版本。仅列与任务有关的设备信息，日志不包含 token 或完整环境变量。

先用 `nvidia-smi` 检查占用。所有设备索引同时记录物理 UUID 和进程逻辑编号；设置 `CUDA_VISIBLE_DEVICES` 后不要仍拿旧物理编号调用进程内 cuda device。

### P0.2 建立隔离运行环境

先读取已有仓库和任何 AGENTS.md，保留用户已有文件。安装最小依赖集合，先模拟器后模型，下载资源使用独立缓存。输出可重建环境锁和实际成功的安装命令；不要只保存 `pip freeze` 而不写 Python/驱动约束。

### P0.3 分别验证 state 和 RGB

按顺序执行单环境 state、少量并行 state、单环境真实相机 RGB、少量并行 RGB。每步保存 exit code、错误摘要、图像 shape/dtype/range、配置和运行时间。

ManiSkill 的视觉渲染要求 Vulkan 配置。[S1] 对 H800 是否可用以本机测试为准；不要只根据 `torch.cuda.is_available()` 判定渲染成功。策略图像从 reset/step 的传感器观测提取，viewer 图像只用于演示。

### P0.4 给出明确分支决策

- `STATE_OK + RGB_OK`：继续完整主线。
- `STATE_OK + RGB_BLOCKED`：继续状态采集和评估，视觉相关验收保持未完成；记录需要的资源或管理员动作。
- 模拟器 GPU 路径失败：可用 CPU backend 继续接口和正确性检查，但所有产物标明 backend，不能声称验证 H800 吞吐。
- 仅 H800：RoboLab 不作为安装主线。无模型权重：继续 mock contract 和非 WAM 策略。

完成条件：`docs/STATUS.md` 记录实际能力矩阵和下一步；一个失败不导致放弃所有独立工作。

## 4. P1：环境、观测与策略接口

### P1.1 固化任务配置

每个任务写 YAML：backend、env_id、robot_id、obs_mode、control_mode、reward_mode、num_envs、episode horizon、control frequency、sim frequency、camera 配置、图像分辨率、seed manifest、split、后端版本。

初始 state 和 RGB 各保留一个配置。调试 RGB 可暂用低分辨率；真实 WAM 接入必须采用其预处理要求，改变分辨率需单独 run，不混入同组结果。

`num_envs` 从 1、4、16 递增，通过测量再增加。不要默认 256 个 RGB 环境。

### P1.2 统一 EnvAdapter

对外提供 reset、step、close 和 specs。保留原始 backend observation，同时给出标准化的 policy observation。不要为追求统一删除后端字段。

`StepBatch` 至少包含：obs_before、action_executed、obs_after_before_reset、reward_env、terminated、truncated、success、episode_id、env_slot、step_index、dt_sim。重置后的 initial observation 必须是另一字段或另一个事件。

状态特权信息仅供诊断和 state 控制基线；不能偷偷放入视觉 WAM 的输入或正式 reward。

### P1.3 统一 PolicyAdapter

建议协议：

```python
class PolicyAdapter:
    def specs(self) -> dict: ...
    def reset(self, env_ids, episode_ids) -> None: ...
    def act(self, observation_batch, request_context) -> "PolicyOutput": ...
    def close(self) -> None: ...
```

`PolicyOutput` 返回 actions `[B, K, D]`、action_spec_id、policy_version、request_id，可选 predictions、prediction_spec、policy_state 和训练 extras。B 为本次请求的环境数；K 为计划动作块长度；D 必须来自动作规范，不能在通用代码写死。

实现 RandomPolicy、合法保持动作的控制策略和 ReplayPolicy。零向量不一定代表“保持不动”，必须根据 controller 验证。另建标记为 `synthetic=true` 的 MockWAM，仅用于测试预测字段、延迟、超时和 reset；输出不进入模型效果图。

### P1.4 独立验收

两个真实任务各完成至少 10 个完整 episode，至少一条动作序列使机器人可观察地运动。保存小型传感器帧样本、动作范围和 episode 日志。RGB 不可用时该部分明确待办。

## 5. P2：Rollout collector 与轨迹格式

### P2.1 正确执行动作块

一次请求给出 K 个动作，配置 `execute_horizon=H`，满足 `1 <= H <= K`；执行 H 个后重新观测、重新规划。环境在块内结束时，清空该环境剩余动作和策略历史。不同 env_slot 不共享 action cursor 或 history。

同时保存 model 原始动作、解归一化动作和真正送入环境的动作。任何 clipping、坐标变换、插值、频率转换必须写入 metadata，不可静默处理。真实 WAM 的转换规则未确认时直接拒绝执行该适配器。

### P2.2 区分终止、截断与自动重置

训练 collector 与 evaluation runner 使用分开的 termination 配置。分别保存 `terminated` 和 `truncated`，不要压成唯一 done 后丢失语义。训练时哪些情况 bootstrap，由训练器定义，采集器提供足够信息，不自行猜测。

重点检查自动 reset 返回的观测是否已经属于新 episode。应提取上一个 episode 的 terminal observation；字段结构以锁定后端版本为准，不机械假设 Gymnasium 和 ManiSkill wrapper 完全一致。

### P2.3 轨迹落盘设计

最小方案：每个 episode 一个 HDF5 文件或有限大小的 HDF5 shard，加 JSON manifest；不用一开始引入数据库。复用官方 recorder 时补齐本项目字段，不同时维护两份含义不同的真值。

| 类别 | 必备字段 |
|---|---|
| 身份 | schema_version、run_id、episode_id、env_slot、task_id、split、seed、policy_version |
| 序列 | obs[0:T+1]、action_raw[0:T]、action_executed[0:T]、reward_env[0:T]、terminated、truncated、success |
| 预测 | request_id、origin_step、预测 offset/timestamp、predicted RGB 或 latent 引用、prediction_valid_mask |
| 复现 | resolved config、代码 commit、依赖锁 hash、checkpoint hash/revision、robot/control/camera spec、initial_state 或足以复现的引用 |
| 审计 | n_transitions、complete、结束原因、重试记录、文件 checksum、policy synthetic/privileged 标记 |

obs 可拆为 rgb、proprio、诊断 state 等组。T+1 的长度规则仅针对完整连续 episode；边界 transition 的 obs_after 必须来自同一 episode。图像只存一次，不为每个 transition 重复存 current/next 副本。

训练 extras 如 old_logprob、value、action mask 仅在真实实现可提供且有明确概率语义时保存；缺少应为 null/absent，绝不能用零值伪造。采集器可供 RL 使用，不等于已经具备 WAM 的 on-policy RL 所需全部统计量。

### P2.4 中断恢复与验证

写入使用临时文件、完整写入后原子重命名。恢复时依据 manifest 的 episode_id、文件校验和和 complete 标记识别完整回合；损坏/未完成回合隔离或从头重跑，默认不承诺任意物理步精确续跑。

实现 `validate_trajectories`：检查 shape、dtype、有限值、action spec、时序、跨回合链接和唯一 ID。保存初始状态也不保证跨模拟器版本的逐步轨迹完全相同；重放使用相同版本和配置并报告误差。

### P2.5 必须覆盖的正确性风险

| 场景 | 期望行为 |
|---|---|
| 向量环境一个 slot 提前结束 | 其他 slot 的历史、动作和 episode_id 不变 |
| 步数上限导致截断 | terminal obs 被保存，truncated 与 terminated 分开 |
| 块内 reset 或实际只执行 H<K | 剩余动作失效，未执行未来帧不被当作匹配真值 |
| policy 超时/NaN/错误维度 | 标记基础设施错误，停止该回合并记录；不暗中替换随机策略 |
| 写文件中断并重启 | 无重复完整 episode，不统计半条轨迹 |

验收：先用可控小型 fake env 精确验证边界，再用真实 simulator 采集两任务各 20 个完整 episode。假环境测试是正确性测试，不是机器人实验结果。

## 6. P3：评估协议与结果生成

### P3.1 固定评估任务和初始条件

在训练和看正式结果之前建立 manifests。基础开发配置：每任务 20 个 validation episodes、50 个 test episodes；独立随机策略的 10/20 个 smoke episodes不作为正式 test。正式团队实验可在预算允许时扩到每任务每 checkpoint 100 个 test episodes。

每个 episode 记录独立 ID、seed、场景/对象参数或初始状态。train/validation/test 的 episode seed 和场景记录不重叠；仅随机种子不同称为 held-out initial conditions，不称为 held-out tasks。真正任务泛化必须整类任务未参与训练。

不同并行数可能改变 RNG 消耗和初始状态序列，因此“相同 seed”不足以证明比较公平。能保存状态就验证恢复；否则固定 num_envs、batch 排序和重配置规则，并记录限制。正式比较不因为模型快慢而给不同 episode 数。

### P3.2 统一完整 episode 评估

ManiSkill 正式评估遵循锁定版本官方方案：不因中途 success/fail 提前 reset，运行到共同 horizon，记录 success_once 和 success_at_end；对象随机化配置按官方建议处理。[S3] 不把这种评估设定强加给训练 collector 或另一个 backend。

默认主指标先设 success_once，success_at_end 同时报告，正式组内实验前冻结；可因具体任务调整，但不能看结果后挑更好看的指标。

| 指标 | 定义与注意点 |
|---|---|
| success_once | 完整回合中至少一次任务成功 |
| success_at_end | 最后有效步任务是否成功 |
| return_env | 仅环境任务奖励之和，供方法间比较 |
| steps / time_to_first_success | 步数与首次成功时间；未成功用 null，不用 0 |
| error_rate | 超时、崩溃、非法动作等基础设施异常比例 |

辅助奖励 `reward_aux`、组合训练奖励 `reward_total` 另存；不能用带不同奖励权重的 total return 证明任务能力更强。

### P3.3 统计与公平对照

正式待接入方法：pretrained WAM（未 RL）、task-only RL、task+alignment RL。当前没有真实 checkpoint 的行标记 `NOT_RUN`，不得填充 mock 数据。

输出每任务成功数/样本数和二项比例 Wilson 95% 区间；跨任务默认 macro average，另附各任务样本数。该区间仅反映给定 checkpoint 下 episode 的有限样本不确定性，不代表训练种子方差。

团队预算允许时建议 3 个独立 training seeds，报告每个 seed 的结果和均值/标准差。一个 checkpoint 配 3 组环境 seeds不等于 3 次独立训练。正式 test 不用于 checkpoint 选择、奖励权重调整或频繁调参。

方法比较固定任务、环境、控制频率、执行动作块长度、相机处理、评估预算；训练方法需同时记录 environment interactions 和 wall time。执行 horizon 变化属于消融，不混入主对照。

运行错误不能静默剔除或只重跑至成功：保存全部尝试；按预先固定的重试次数处理基础设施错误，报告 intended/completed/failed 数量。缺失 episode 的正式结果应标记 incomplete，必要时另列按失败处理的保守结果。

### P3.4 输出文件

每个 run 包含 `config.resolved.yaml`、`provenance.json`、`episodes.csv`、`summary.json`、`errors.jsonl`、`manifest.json`。汇总工具输出 per-task success 图、end-success 图和吞吐表。无训练日志时不生成“训练收敛曲线”。

验收：给定同一 episode 表，重新汇总得到相同结果；人工构造“中途成功最后失败”的 episode 必须输出 success_once=1、success_at_end=0。检查汇总分母及跨任务权重。

## 7. P4：可用控制策略与资源测量

### P4.1 获得一个正向控制

优先利用锁定 ManiSkill 版本提供的同任务演示/官方 motion planning 或现成策略，获得至少一个真实成功回合，并明确使用了哪些特权状态。不要手写大量 IK/运动规划来证明框架能用。

若演示不可用，可选跑官方 state-based PPO 小基线，但这只是 infrastructure control，不是 WAM RL baseline，更不能替代 Proposal 的核心对照。首次上限 2 GPU-hours，到点保存 checkpoint 和实际成功率；不保证在该时间内收敛，也不为了凑结果无限训练。

若无法得到真实成功回合，边界逻辑可由 synthetic fixture 验证，但真实正向控制仍标记 `PENDING`。

### P4.2 测量小型吞吐矩阵

state 和 RGB 分开，num_envs 取 1、4、16，GPU memory 允许且吞吐确有收益时再试 32/64。每组先 warmup 50 steps，再测 500 steps，重复 3 次。慢模型接入后可以改为固定测量时长，须记录变更原因。

记录 transitions/s、episodes/hour、policy latency p50/p95、env step、render、serialization/write、peak allocated/reserved VRAM、进程总显存、RSS、写盘字节与 wall time。计时区间明确 GPU 异步与同步边界；组件时间与端到端耗时分别报告，不能把重叠时间直接相加。

分别测不开录像的实际采集与开启录像的演示配置。`vector steps/s × num_envs` 只对有效 transition 计数成立，不能把 reset/无效 padding 算成数据。

### P4.3 估算后续成本

用实测 episodes/hour 估算评估耗时，并单独计入模型加载、重置、落盘与 warmup。双卡只在独立进程并发确实提高总吞吐时采用，不假设线性加速。

原始单相机 RGB 数据量近似：E × (T+1) × H × W × 3 bytes。例：100 回合、200 动作、128×128 RGB，约 0.99 GB（十进制），不含其他数据；256×256 约 3.95 GB，多相机相乘。先保存小集，测压缩比再扩大。MP4 可用于演示，不作为有损像素奖励的唯一真值来源。

完成条件：一个实测推荐配置、显存余量、每 100 episodes 预算和已知瓶颈。当前目标不是优化 CUDA kernel。

## 8. P5：交接接口，不等待队友即可完成

### P5.1 与王俊昌的模型接口

交付 `docs/INTERFACE_CONTRACT.md`，其中明确版本号和下列字段。当前未确认项写 UNKNOWN，不能猜。

| 接口类别 | 必须确认 |
|---|---|
| 输入 | 相机数量/顺序、RGB/BGR、shape/range、历史帧、proprio、instruction、goal 是否必需 |
| 动作 | D、每维含义、单位、绝对/增量、旋转约定、坐标系、归一化统计、夹爪方向、控制频率 |
| 时序 | K、H、是否闭环重规划、每预测帧对应的物理时间、是否含当前观测帧 |
| 输出 | actions、可选真实预测视频/latent、checkpoint revision、episode reset、batch 能力 |
| 训练 | 在线还是离线采样、policy version、可计算概率/梯度的路径、训练 extras 的实际语义 |

DROID→ManiSkill 的语义转换归属需双方确认。你可以实现适配骨架与检验；由模型负责人确认语义，不因 tensor shape 对上就认定机器人动作正确。

### P5.2 预测与真实状态配对

每个预测绑定 `(episode_id, request_id, origin_step, action_sequence_hash, prediction_offset_time)`。真实图像带相机 ID、仿真时间、控制步号及预处理版本。

若一次预测未来 K 步、实际仅执行 H 步，那么 H 之后的真实轨迹可能已经由新策略动作产生；不能继续用旧 prediction 做奖励，除非方法明确建模新的条件。遇到 reset、drop frame、插值、动作 clipping 或频率不匹配，标记 validity 和原因。

记录预测视图与真实相机视图是否匹配；如果只能输出 latent，保存 encoder/version 和空间含义，不能声称已经完成像素误差比较。缺失预测返回 unavailable，不填一张空白图伪装真实预测。

### P5.3 Reward hook 与训练交接

提供只接收已完成 transition/片段的 `RewardHook.compute(batch)`，返回 auxiliary reward、valid mask、diagnostics、reward version。采集器保存环境奖励及原始预测，组合方式由训练配置控制。

允许零辅助奖励 hook 和明确标记的 synthetic hook 检查管线。是否使用 MSE、特征误差、归一化或潜势奖励交给奖励负责人。训练器如需反向传播 through world model，不能使用离线导出或 inference-only HTTP 当成可微计算图。

模型可能不提供常规 action logprob；不得随意套 PPO 或造 old_logprobs。训练负责人需决定算法、采样粒度、off-policy/on-policy 条件和更新规则。

### P5.4 与王少航的实验接口

提供单实验命令、配置 schema、退出码、run manifest 和可机器读取的 summary。实验管理同学据此组织奖励权重/训练模块消融；本模块不建设另一个复杂 scheduler。

从小数据中交付：真实 state 轨迹、若可用则真实 RGB 轨迹、episode 汇总示例、synthetic 模型接口 fixture。四者来源和用途分开标注。

## 9. P6：需要队友或额外资源后才能完成

按以下顺序进行，每步过关才扩大：

1. 真实 WAM 对一份离线观测成功推理，核对动作语义和预测时间。
2. 在一个兼容仿真任务上闭环执行一个 episode，再扩到 10 个；成功率为零时先检查接口和 domain mismatch，不能直接归因于 RL。
3. 同一 checkpoint 跑统一评估，保存完整来源与失败案例。
4. 接入 task-only RL checkpoint，再接入辅助奖励版本；确认 train/test 未泄漏。
5. 按冻结协议执行最终比较，交给实验分析负责人汇总。

如果 WAM 无法在 ManiSkill 使用，选择现成兼容的 RoboLab+RTX 路径需团队确认资源；保留已完成的通用 schema、评估和采集代码，不重复建设。若换模型/任务改变研究问题，记录在 deviations.md 并由团队确认，不静默偏离 Proposal。

## 10. 建议仓库结构

以下是目标结构，初始仓库目前仅有本计划及 README。不要为了形式创建大量空目录；按阶段实现。

```text
README.md
pyproject.toml
docs/
  RENJIE_EXECUTION_PLAN.md
  STATUS.md
  INTERFACE_CONTRACT.md
  EVALUATION_PROTOCOL.md
  RUNBOOK_H800.md
  deviations.md
configs/
  envs/
  policies/
  eval/
  benchmarks/
manifests/
  validation.jsonl
  test.jsonl
src/robots_project/
  envs/
  policies/
  rollout/
  data/
  evaluation/
  cli/
tests/
  test_episode_boundaries.py
  test_action_chunks.py
  test_trajectory_integrity.py
  test_evaluation_metrics.py
  test_prediction_alignment.py
scripts/
outputs/                 # gitignored
data/                    # gitignored
checkpoints/             # gitignored
```

所有本机目录可由环境变量或 CLI 设置，但不要使用系统 HOME/CODEX_HOME 作为项目临时变量。上游源码保留 commit 引用，原则上不整仓复制到本项目。

### 10.1 待实现 CLI 的验收形状

以下不是已经存在的命令。Codex 应实现这些能力，若采用其他路径，在 RUNBOOK 中更新可实际执行的命令。

```bash
python -m robots_project.cli.preflight --output outputs/preflight.json
python -m robots_project.cli.smoke --config configs/envs/pushcube_state.yaml
python -m robots_project.cli.collect --config configs/envs/pushcube_rgb.yaml --policy random --episodes 20
python -m robots_project.cli.validate_trajectories --run-dir outputs/RUN_ID
python -m robots_project.cli.evaluate --config configs/eval/dev.yaml --policy random
python -m robots_project.cli.summarize --run-dir outputs/RUN_ID
python -m robots_project.cli.benchmark --config configs/benchmarks/h800.yaml
```

命令必须有 `--help`，config validation 和明确非零失败退出码；不通过吞掉异常令脚本“看起来成功”。

## 11. 验收清单

### 11.1 独立部分完成条件

- [ ] P0：真实机器预检和依赖锁可复现；state/RGB 能力分开标记。
- [ ] P1–P2：两个真实任务可执行，轨迹 schema 与边界测试通过，恢复无重复。
- [ ] P3：固定评估协议、episode 表和统计结果；错误和未完成回合可审计。
- [ ] P4：实际吞吐/显存/磁盘预算，至少一个真实正向控制或明确待补状态。
- [ ] P5：接口规范、示例数据、运行说明和未解决问题可交接。

只有 state 部分完成时应标记“状态路径已验收，视觉路径待完成”，不能勾选完整视觉基础设施。

### 11.2 完整团队项目仍需的证据

- [ ] 真 WAM 与 simulator 观测/动作/时间对齐。
- [ ] 实际 RL 更新和可用 checkpoint，不仅是 rollout server。
- [ ] task-only 与 auxiliary reward 的受控对照及必要消融。
- [ ] 真实训练稳定性、样本效率和任务效果结果；无效果也如实报告。
- [ ] 教师 rubric 要求的最终报告、展示和其他交付。

## 12. 常见阻塞与停止条件

| 阻塞 | 当前可继续的工作 | 不能做的替代 |
|---|---|---|
| RoboLab 需要 RTX，机器仅 H800 | ManiSkill、schema、评估；准备 RTX 接入文档 | 用 CUDA 可用冒充 RoboLab 可用 |
| Vulkan/RGB 失败 | state、单元测试、日志/统计；记录具体错误 | 用黑图/随机图冒充视觉轨迹 |
| 队友模型未交付 | mock contract、Random/ReplayPolicy、边界测试 | 把 mock 效果写成 WAM 结果 |
| 没有真实成功回合 | 查动作/回放，保留 synthetic 指标测试 | 改任务成功函数使结果变好 |
| 缺少 logprob 或训练梯度路径 | 保存原始采样数据，登记训练接口缺口 | 生成零 logprob 并宣称 PPO 可用 |

同一环境问题排查超过 2 小时且无新证据，写入 BLOCKERS，换到不依赖它的模块。用户已授权的低风险本地实现不反复询问；涉及系统驱动替换、占用未指定机器或实质改变研究范围时，先给具体问题和可执行替代。

## 13. 给 H800 上 Codex 的执行指令

直接复制以下内容到已进入仓库的 Codex 会话：

> 阅读 README.md 和 docs/RENJIE_EXECUTION_PLAN.md，检查仓库当前状态与已有 AGENTS.md，完成蔡仁杰负责的仿真环境、rollout infrastructure 和 evaluation 的独立部分。执行 P0–P5，按验收结果推进，不只给建议或创建占位代码。
>
> 先诊断真实 H800、驱动、隔离 Python 环境和相机渲染。仅 H800 时不要把需要 RTX 的 RoboLab 当默认可运行；以 ManiSkill 为主线。state 和 RGB 分别验收。先单卡、小任务、少量环境，再测吞吐。默认初期累计预算 12 GPU-hours，单个验证训练不超过 2 小时，不打断其他 GPU 进程。
>
> 实现策略无关接口、动作块执行、正确的 terminal observation/auto-reset 处理、可恢复轨迹保存和统一评估。对 episode 边界、部分 reset、H<K 预测对齐、统计分母和中断恢复做有意义的测试。使用真实模拟器证据验证，synthetic/mock 只测试接口，绝不作为 WAM 或 RL 实验结果。
>
> 不代写队友的奖励研究或默认为自己接管 RL 训练器。不确定的动作语义、模型预测字段和 logprob 能力写入 INTERFACE_CONTRACT.md；先完成不依赖这些信息的工作。下载或真实模型缺失时继续独立部分，不无限等待。
>
> 每个阶段将命令、exit code、结果路径、版本和验收状态写入 docs/STATUS.md；生成 RUNBOOK_H800.md、EVALUATION_PROTOCOL.md、INTERFACE_CONTRACT.md 和 deviations.md。完成后报告已通过、部分通过、未运行及下一步依赖。按阶段做小 commit；保留用户变更，不上传模型、原始轨迹、大视频或密钥。不要凭本指令自动推送未经当前会话授权的新增远端变更。

## 14. 来源与事实边界

核心分工来自用户上传的 `RL WAM.pdf`（一页 Project Proposal），没有将原 PDF 自动上传到公开仓库。下列是技术参考，不是该课程评分标准；执行时记录实际使用的上游版本。

- [S1] ManiSkill installation / Vulkan：https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- [S2] ManiSkill quickstart：https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/getting_started/quickstart.md
- [S3] ManiSkill RL setup / evaluation：https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/reinforcement_learning/setup.md
- [S4] RoboLab README / hardware：https://github.com/NVlabs/RoboLab#requirements
- [S5] NVIDIA Cosmos policy server and RoboLab client：https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/action/run_policy_with_cosmos_framework.md
- [S6] Cosmos3-Edge-Policy-DROID model card：https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID
- [S7] V-JEPA 2 / V-JEPA 2-AC official repository：https://github.com/facebookresearch/vjepa2

任务规模、时间区间、资源上限、目录、协议字段和验收阈值均为本计划提出的工程建议；不是论文报告值或已完成测量。Proposal 的模型训练时间、任务全集和泛化承诺仍需实际验证。
