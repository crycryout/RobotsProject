"""Render classroom figures from verified CSV tables and two real endpoint frames."""
import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image

OUT=Path(__file__).resolve().parents[1]
F=OUT/"figures"
FONT=os.environ.get("CJK_FONT","/home/renjie/qwen_pcie/fonts/NotoSansSC-400.ttf")
font_manager.fontManager.addfont(FONT)
mpl.rcParams.update({"font.family":font_manager.FontProperties(fname=FONT).get_name(),
                     "pdf.fonttype":42,"axes.unicode_minus":False,"axes.spines.top":False,
                     "axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
INK="#142e3a"
MUTED="#455f6a"
GRID="#dae4e7"
COL={"T′":"#254b63","A-video":"#087c76","A-action":"#b55c2c","Base":"#77848b"}


def read(n):
    with (OUT/"data"/n).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save(fig,name):
    fig.savefig(F/(name+".png"),dpi=210)
    fig.savefig(F/(name+".pdf"))
    plt.close(fig)


def flow():
    fig=plt.figure(figsize=(12.4,4.25))
    ax=fig.add_axes([0,0,1,1],xlim=(0,12.4),ylim=(0,4.25));ax.axis("off")
    def box(x,y,w,h,title,body,fill="#f0f5f6",edge="#b7cbd0"):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.09",
                                    linewidth=1.0,edgecolor=edge,facecolor=fill))
        ax.text(x+.13,y+h-.22,title,fontsize=14,color=INK,va="top",weight="bold")
        ax.text(x+.13,y+h-.62,body,fontsize=10.2,color=MUTED,va="top",linespacing=1.5)
    def arrow(x1,y1,x2,y2):
        ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",lw=1.5,color="#51727c",mutation_scale=13))
    box(.10,2.05,2.46,1.45,"RoboLab 重置","任务、提示与 seed\n12 个随机化起点")
    box(2.93,2.05,2.61,1.45,"策略观测","腕部与双肩相机 RGB\n7 个关节、夹爪、语言指令")
    box(5.92,2.05,2.70,1.45,"Cosmos3-Edge","联合生成未来视频潜变量\n及 32 × 8 动作块")
    box(9.00,2.05,3.21,1.45,"执行与终止","15 Hz，每控制步执行一行\n任务成功或到时即冻结该环境")
    for x1,x2 in [(2.57,2.92),(5.55,5.91),(8.64,8.99)]:
        arrow(x1,2.78,x2,2.78)
    ax.plot([10.65,10.65,4.23],[3.52,3.86,3.86],c="#51727c",lw=1.3)
    arrow(4.23,3.86,4.23,3.51)
    ax.text(7.25,3.93,"仍在运行的环境在块边界重新观测",ha="center",fontsize=10.5,color=MUTED)
    box(.40,.10,5.73,1.18,"训练采集","真实第 16 步画面与预测对齐，任务进度与成功产生奖励",fill="#eaf5f1",edge="#b6d7cb")
    box(6.49,.10,5.72,1.18,"固定策略评测","读仿真成功标记，按任务、seed 与环境编号配对",fill="#edf2f7",edge="#becfdc")
    ax.plot([10.65,10.65,3.15],[2.04,1.65,1.65],c="#51727c",lw=1.3)
    arrow(3.15,1.65,3.15,1.30);arrow(9.34,1.65,9.34,1.30)
    save(fig,"rollout_flow")


def training():
    fig,ax=plt.subplots(figsize=(10.5,3.8))
    fig.subplots_adjust(left=.09,right=.99,bottom=.18,top=.76)
    fig.text(.04,.93,"三轮训练采样成功率",fontsize=17,color=INK,weight="bold",va="center")
    fig.text(.04,.825,"各臂一个训练 seed；每点 96 个训练回合，含 default 和 specific，采集间更新策略。",fontsize=10,color=MUTED)
    rows=read("training_epochs.csv")
    for model,marker,off in [("T′","o",-0.018),("A-video","s",0),("A-action","^",.018)]:
        vals=[r for r in rows if r["model"]==model]
        x=np.array([int(r["epoch"]) for r in vals])+off
        y=[100*int(r["success"])/int(r["episodes"]) for r in vals]
        ax.plot(x,y,marker=marker,c=COL[model],ms=8,lw=2.4,label=model,zorder=4,alpha=.95)
        for xp,yp in zip(x,y):
            if model=="T′" and xp<1: continue
            dy=8 if model=="A-video" else -16 if model=="A-action" or xp>2.5 else 8
            if model=="A-action" and xp>2.5: dy=7
            ax.annotate(f"{yp:.1f}%",(xp,yp),xytext=(0,dy),textcoords="offset points",ha="center",fontsize=9.5,color=COL[model])
    ax.set(xlim=(.74,3.26),ylim=(35,62),xticks=[1,2,3],xticklabels=["第 1 轮","第 2 轮","第 3 轮"],
           yticks=[35,40,45,50,55,60]);ax.set_yticklabels(["35%","40%","45%","50%","55%","60%"])
    ax.grid(axis="y",c=GRID);ax.set_axisbelow(True)
    for side in ("left","bottom"):ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED,labelsize=11)
    fig.legend(loc="upper right",bbox_to_anchor=(.99,.99),ncol=3,frameon=False,fontsize=11)
    ax.text(.99,.02,"第 1 轮 T′ 与 A-video 均为 48/96",transform=ax.transAxes,color=MUTED,ha="right",fontsize=8.4)
    save(fig,"training_curve")


def training_steps():
    rows = read("training_collections.csv")
    times = read("training_runtime.csv")
    fig, ax = plt.subplots(figsize=(11.3, 4.1))
    fig.subplots_adjust(left=.09, right=.68, top=.76, bottom=.24)
    fig.text(.025,.945,"训练采样成功率随累计环境步数变化",fontsize=16,color=INK,weight="bold",va="center")
    fig.text(.025,.835,"每点为一次采集的 48 回合。三组各用一个训练 seed，均含 default 和 specific 指令。",fontsize=10.5,color=MUTED)
    for runtime,marker in zip(times,("o","s","^")):
        model=runtime["model"]
        points=[r for r in rows if r["model"]==model]
        ax.plot([int(r["cumulative_env_steps"])/10000 for r in points],
                [float(r["success_pct"]) for r in points],color=COL[model],marker=marker,
                markersize=6,linewidth=1.8,label=model)
    ax.set(xlabel="各臂累计环境控制步（万步）",ylabel="本次采集成功率",xlim=(0,14),xticks=range(0,15,2),
           ylim=(20,80),yticks=[20,40,60,80],yticklabels=["20%","40%","60%","80%"])
    ax.grid(axis="y",color=GRID);ax.set_axisbelow(True)
    for side in ("left","bottom"): ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)
    fig.text(.735,.71,"三轮累计步数  /  实际耗时",fontsize=11.5,weight="bold",color=INK)
    for i,r in enumerate(times):
        y=.57-i*.15
        fig.text(.735,y,r["model"],color=COL[r["model"]],fontsize=10.5,weight="bold")
        fig.text(.735,y-.065,f'{int(r["cumulative_env_steps"]):,} 步   ·   {r["elapsed_hms"]}',fontsize=10,color=INK)
    fig.text(.025,.025,"环境步数取各回合实际执行动作数之和。耗时从训练启动至状态记为 done，格式为时:分:秒。",fontsize=9.6,color=MUTED)
    save(fig,"training_env_steps")


def evaluation():
    fig,(a,b)=plt.subplots(1,2,figsize=(11.6,3.9),gridspec_kw={"width_ratios":[1,1.10]})
    fig.subplots_adjust(left=.115,right=.968,bottom=.28,top=.70,wspace=.55)
    results=[r for r in read("results.csv") if r["dataset"]=="untrained_task_eval" and r["status"]=="verified"]
    order=[r["model"] for r in results]
    fig.text(.025,.94,"冻结策略在 10 个未参与 v2 更新的任务上的评测",va="center",fontsize=16,color=INK,weight="bold")
    a.set_title("仿真任务成功率",loc="left",fontsize=12,color=INK,pad=12)
    for i,r in enumerate(results):
        y=len(results)-1-i
        val=float(r["success_pct"]);a.barh(y,val,color=COL[r["model"]],height=.42)
        a.text(val+2,y,f"{r['success']}/{r['episodes']}  ·  {val:.1f}%",va="center",fontsize=10,color=INK)
    a.set(xlim=(0,100),yticks=list(reversed(range(len(order)))),yticklabels=order,xticks=[0,25,50,75,100])
    a.set_xticklabels(["0%","25%","50%","75%","100%"])
    b.set_title("相对 T′ 的配对差值与 95% 区间",loc="left",fontsize=12,color=INK,pad=12)
    pairs=[r for r in read("paired.csv") if r["reference"]=="T′"]
    b.axvline(0,color="#74858b",lw=1.2,ls="--")
    for y,r in zip([1,0],pairs):
        avg,lo,hi=map(float,(r["delta_pp"],r["ci_low_pp"],r["ci_high_pp"]))
        b.errorbar(avg,y,xerr=[[avg-lo],[hi-avg]],c=COL[r["model"]],fmt="o",capsize=5,ms=8,lw=2.2)
        b.text(avg,y+.19,f"{avg:+.1f}  [{lo:+.1f}, {hi:+.1f}]",color=COL[r["model"]],fontsize=10,ha="center")
    b.set(ylim=(-.45,1.55),xlim=(-11,10),yticks=[1,0],yticklabels=["A-video","A-action"],xticks=[-10,-5,0,5,10])
    b.set_xlabel("成功率差值（百分点）",fontsize=10,color=MUTED)
    for ax in (a,b):
        ax.grid(axis="x",color=GRID);ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False);ax.spines["bottom"].set_color(GRID);ax.tick_params(colors=MUTED,labelsize=10)
    fig.text(.025,.095,"每个训练臂 1 个 seed，e03c1；评测统一 default，10 项来自已筛选 train 候选池。正式 test 另列。",fontsize=9.7,color=MUTED)
    fig.text(.025,.025,"95% 区间在任务内整对重采样。三臂 HDF5 物理初态一致；Base 仅能复核协议键，原始初态轨迹已清理。",fontsize=9.7,color=MUTED)
    save(fig,"independent_eval")


def cases():
    rows=read("eval_episodes.csv")
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.8))
    fig.subplots_adjust(left=.02,right=.98,top=.86,bottom=.02,wspace=.08)
    for ax,env,title in zip(axes,(0,5),("成功 · 第 150 步","超时 · 第 600 步")):
        row=next(r for r in rows if r["model"]=="T′" and r["task"]=="BananaOnPlateTask" and int(r["env"])==env)
        p=Path(row["source"]);log=json.loads(p.read_text())
        im=np.load(p.parents[2]/"capture"/log["v1_progress"]["terminal_frame"]["file"],allow_pickle=False)
        # Display both low-mounted side views, sampled from the same terminal composite.
        ax.imshow(im[360:540]);ax.set_title(f"env {env}   |   {title}",fontsize=13,color=INK,pad=12);ax.axis("off")
    fig.text(.5,.06,"T′ 最终策略 · 香蕉放盘 · 同任务两个起点 · 左右肩部相机终帧",ha="center",fontsize=11,color=MUTED)
    save(fig,"case_endpoints")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("figures",nargs="*",help="Optional subset: flow, training, evaluation, cases")
    selected=parser.parse_args().figures or ["flow","training","evaluation","cases"]
    renders={"flow":[flow],"training":[training,training_steps],"evaluation":[evaluation],"cases":[cases]}
    if not all(name in renders for name in selected):
        parser.error("Unknown figure selection")
    F.mkdir(exist_ok=True)
    for name in selected:
        for render in renders[name]:
            render()
