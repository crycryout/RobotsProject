"""Derive the presentation table. Keep direct and historically inferred base comparisons labelled."""
import csv
import json
from pathlib import Path
from report_scope import EVAL, TRAIN

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"data"
def read(name):
    with (D/name).open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def pct(x):
    return f"{float(x):.1f}%"
def signed(x):
    return f"{float(x):+.1f}"
def main():
    rows=read("results.csv")
    cross=json.loads((D/"base_crosscheck.json").read_text())
    base=next(r for r in rows if r["dataset"]=="untrained_task_eval" and r["model"]=="Base")
    direct=base["status"]=="verified"
    base_count=int(base["success"]) if direct else int(cross["implied_base_success"])
    t_ref=next(r for r in rows if r["dataset"]=="untrained_task_eval" and r["model"]=="T′")
    t_rate=100*int(t_ref["success"])/int(t_ref["episodes"])
    models=["Base","T′","A-video","A-action"]
    md=["# 结果表｜训练采集与固定策略评测","",
        "每个 v2 训练臂使用同一个训练 seed（20261123），前三轮训练采样含 default 和 specific。固定模型评测仅用 default，取已筛选的原 train 候选池中未参与 v2 更新的 10 项任务，各 12 个起点。正式 test split 另行划分，本次未测。所有差值以百分点表示。","",
        "| 模型 | 三轮训练采样 | 训练采样相对 T′ | 固定策略在未训练任务上的评测¹ | 相对 T′ | 相对原始 Base² |",
        "|---|---:|---:|---:|---:|---:|"]
    display=[]
    for m in models:
        t=next(r for r in rows if r["model"]==m and r["dataset"]=="training_rollout")
        e=next(r for r in rows if r["model"]==m and r["dataset"]=="untrained_task_eval")
        successes=int(e["success"]) if e["success"]!="" else base_count
        e_rate=100*successes/120
        grade="逐回合重算" if e["status"]=="verified" else "旧文档与旧回合间接核算"
        train=f"{t['success']}/{t['episodes']} · {pct(t['success_pct'])}" if t["success"]!="" else "不适用"
        train_diff=signed(t['delta_vs_T_pp']) if t['delta_vs_T_pp']!="" else "—"
        target=signed(e_rate-t_rate)
        base_diff=signed(e_rate-100*base_count/120)
        md.append(f"| {m} | {train} | {train_diff} | {successes}/120 · {e_rate:.1f}%{'（间接）' if e['status']!='verified' else ''} | {target}{'（间接）' if e['status']!='verified' else ''} | {base_diff}{'（间接）' if not direct else ''} |")
        display.append(dict(model=m,training_success=t["success"],training_episodes=t["episodes"],training_pct=t["success_pct"],
                            independent_success=successes,independent_episodes=120,independent_pct=e_rate,
                            independent_evidence=grade,delta_vs_T_pp=float(target),delta_vs_base_pp=float(base_diff),
                            delta_vs_base_evidence="逐回合协议键匹配，Base 无物理初态原件" if direct else "间接总量，配对区间待核",
                            v2_training_seed=20261123,training_seeds_per_v2_arm=1,**TRAIN,**EVAL))
    with (D/"report_table.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=display[0]);w.writeheader();w.writerows(display)
    md+=["","¹ Base 取冻结预训练模型的筛选回合，三个训练臂固定第三轮最后一次更新的 e03c1。每种策略各有 120 个评测回合。训练过程中先采集数据，再更新得到新的 checkpoint。","",
       "² 原始 Base 的直接筛选表尚未取得。旧 v1 e02c2 在同一批物理起点成功 50/120，项目文档列出 v1 − Base = −0.008（保留三位小数，n=120）。整数净差只能是 −1，因此可间接核算 Base 总数为 51。所有相对 Base 的百分比差只使用这个间接总量，逐任务分布、成败交换和置信区间须等待原表。直接证据 CSV 中用空值保留这项缺口。[筛选归档位置与接收命令](evidence/共享路径核查.md)"
       if not direct else "² Base 成绩直接来自 10 份原始筛选 Parquet。26 项候选共 312 条 default 回合，只按预定队列计入 120 条。四种策略按任务、提示、批次、seed 和环境编号配对。三组 e03c1 另有 HDF5 物理初态逐项比对，Base 的原始 HDF5 未随筛选表保留。",
       "","[汇报表 CSV](data/report_table.csv) · [直接证据 CSV](data/results.csv) · [旧 v1 汇总的独立交叉核对](data/base_crosscheck.json)",
       "","## 120 个配对起点","","“胜”表示该模型成功而参照失败。区间针对固定十项任务，任务内重采样整对回合 20,000 次。双侧精确 McNemar 检验使用胜负不一致的回合，p 值为未经多重比较调整的描述性结果。","",
       "| 模型减参照 | 胜 / 负 / 同 | 差值（百分点） | 95% 配对区间（百分点） | 精确 p |",
       "|---|---:|---:|---:|---:|"]
    for p in read("paired.csv"):
        md.append(f"| {p['model']} − {p['reference']} | {p['win']} / {p['loss']} / {p['tie']} | {signed(p['delta_pp'])} | [{signed(p['ci_low_pp'])}, {signed(p['ci_high_pp'])}] | {float(p['exact_mcnemar_p']):.3f} |")
    md+=["","[配对数据](data/paired.csv) · [回合键、版本、初态哈希与原始 JSON 路径](data/eval_episodes.csv)",
         "","## 全部十个评测任务","","每格为成功数 / 12，0/12 表示该策略在该任务的 12 个起点均未完成目标。","",
         "| RoboLab 任务 | 原始 Base | T′ | A-video | A-action |","|---|---:|---:|---:|---:|"]
    per=read("eval_tasks.csv")
    for t in sorted({r["task"] for r in per}):
        rec={r["model"]:r for r in per if r["task"]==t}
        md.append("| "+t+" | "+" | ".join(f"{rec[m]['success']}/12" if rec[m]["success"]!="" else "待核" for m in models)+" |")
    md+=["","[逐任务百分比与差值 CSV](data/eval_tasks.csv) · [任务、场景、时限与 seed](data/eval_batches.csv)",
         "","## 三轮训练曲线的数据","","每格为成功数 / 96。各臂第一轮内已发生一次策略更新。","",
         "| epoch | T′ | A-video | A-action |","|---|---:|---:|---:|"]
    e=read("training_epochs.csv")
    for i in (1,2,3):
        r={x["model"]:x for x in e if int(x["epoch"])==i}
        md.append("| "+str(i)+" | "+" | ".join(f"{r[m]['success']}/96" for m in models[1:])+" |")
    md+=["","[训练逐回合](data/training_episodes.csv) · [按训练任务及提示](data/training_tasks.csv) · [原始记录压缩包](data/source_records.zip)",
         "","## 实际训练步数与耗时","","每臂 6 次采集，每次 48 回合。累计步数是所有环境实际执行动作数之和，耗时取训练启动至任务状态写入 done 的时间差。","","| 训练臂 | 累计环境步数 | 实际耗时（时:分:秒） |","|---|---:|---:|"]
    for r in read("training_runtime.csv"):
        md.append(f"| {r['model']} | {int(r['cumulative_env_steps']):,} | {r['elapsed_hms']} |")
    md+=["","A-video 第 4–6 轮仅有团队汇总，缺少同口径逐次步数及同期 T′。这些数据不足以推断收敛、样本效率或跨 seed 稳定性。[逐次采集](data/training_collections.csv) · [起止时间与原文件](data/training_runtime.csv)"]
    (ROOT/"结果表.md").write_text("\n".join(md)+"\n")

if __name__=="__main__":main()
