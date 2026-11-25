"""Build the Chinese presentation, native speaker notes, and the one-page protocol."""
import csv
import json
import os
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

OUT=Path(__file__).resolve().parents[1]
FONT=os.environ.get("CJK_FONT","/home/renjie/qwen_pcie/fonts/NotoSansSC-400.ttf")
pdfmetrics.registerFont(TTFont("CJK",FONT))
INK="#142e3a";MUTED="#46606d"; TEAL="#11776e"; ORANGE="#a6532d"
LIGHT="#ecf5f2"; PALE="#edf2f5"; RULE="#c9d9dc"; WHITE="#ffffff"
W,H=960,540
def rgb(hexcode):return RGBColor.from_string(hexcode[1:])
def txt(x,y,s,size=17,color=INK,w=None):return ("txt",x,y,s,size,color,w)
def rect(x,y,w,h,color):return ("box",x,y,w,h,color)
def pic(x,y,w,h,name):return ("pic",x,y,w,h,str(OUT/"figures"/name))
def headline(page,topic,title,subtitle):
    return [rect(0,0,W,H,WHITE),rect(0,0,12,H,TEAL),txt(41,24,"ROBOTS   /   仿真、ROLLOUT 与评测",11,TEAL),
            txt(42,51,title,30,INK,880),txt(43,96,subtitle,15,MUTED,875),
            rect(42,504,877,1,RULE),txt(43,513,"Cai Renjie   ·   Robots   ·   2026.11.26",9,MUTED),
            txt(280,513,"固定评测范围    单训练 seed  ·  default  ·  已筛选 train 任务，未用于 v2 更新  ·  正式 test 未测",9,MUTED,543),
            txt(842,513,f"{topic}   {page}/5",10,MUTED)]
def note_band(label,line):
    return [rect(44,440,872,49,LIGHT),txt(57,454,label,15,TEAL),txt(164,454,line,15,INK,736)]

def deck():
    slides=[]
    s=headline(1,"环境","从一次观测到一次抓放","同一条轨迹把预测画面、实际执行的动作和仿真判定联系起来。")
    s += [pic(36,126,889,305,"rollout_flow.png")]
    s += note_band("仿真验证","H800 运行 RoboLab 的真实重置与五步动作，环境、布局和客户端共 30 项测试通过。")
    slides.append(s)

    s=headline(2,"协议","公平比较从同一起点开始","四种策略各有一个明确角色。核心比较使用三个训练臂预先固定的 e03c1。")
    s += [rect(44,148,411,264,PALE),txt(61,163,"训练与筛选",19,INK),
          txt(61,200,"5 个任务   /   8 个任务与指令组合",18,TEAL,379),
          txt(61,239,"Base     原始预训练策略，冻结后筛选任务",14,INK),
          txt(61,268,"T′          task-only RL，共用任务奖励",14,INK),
          txt(61,297,"A-video    对齐信号作用于视频偏好",14,INK),
          txt(61,326,"A-action   对齐信号加入动作偏好",14,INK),
          txt(61,375,"各训练臂的顶层 seed 均为 20261123。",12,MUTED),
          rect(477,148,439,264,LIGHT),txt(496,163,"固定模型评测",19,INK),
          txt(496,200,"10 个任务   ×   12 个起点",23,TEAL),
          txt(497,248,"同一 default 指令、批 seed、策略采样规则",15,INK),
          txt(497,280,"同一任务时限，同一官方放置成功条件",15,INK),
          txt(497,312,"三组 e03c1 的 120 份物理初态逐项一致",15,INK),
          txt(497,373,"Base 按协议键匹配，筛选归档未保留其 HDF5 初态。",12,MUTED)]
    s+=note_band("任务范围","10 项取自筛选过的 train 候选池，部分场景与训练共享。正式 test split 留待另测。")
    slides.append(s)

    s=headline(3,"训练","花了多少环境步，采到了多少次成功？","横轴按实际执行的动作累计。共同前三轮各采集 288 回合，A-video 单独续训的三轮另记。")
    s += [pic(48,123,861,312,"training_env_steps.png")]
    s += note_band("三轮合计","T′  144/288（50.0%）     A-video  147/288（51.0%）     A-action  125/288（43.4%）")
    slides.append(s)

    s=headline(4,"评测","在这十个任务上，辅助对齐未显示成功率收益","主比较以只用任务奖励的 T′ 为参照。区间保留每个起点上两个策略的成败配对。")
    s += [pic(35,131,890,299,"independent_eval.png")]
    s+=note_band("逐对看","A-video 有 5 对由败转胜、8 对由胜转败。A-action 为 7 胜、7 负。两个区间都跨零。")
    slides.append(s)

    s=headline(5,"案例","同一策略，两个不同结局","香蕉放盘 · T′ 的采样帧回放与终帧 · 每 32 步采一张，另取终帧。两类各选最小环境编号。")
    s += [pic(41,131,878,309,"case_endpoints.png")]
    s += note_band("结论范围","四种策略共 480 条记录，120 个配对起点。每个训练臂只有一个训练 seed。")
    slides.append(s)
    return slides


def fit(text,size,width):
    if width is None:return size
    while pdfmetrics.stringWidth(text,"CJK",size)>width-1 and size>6:size-=0.25
    return size

def draw_pdf(c,objects,height):
    for op,x,y,*args in objects:
        if op=="txt":
            s,size,color,width=args;size=fit(s,size,width)
            c.setFillColor(colors.HexColor(color));c.setFont("CJK",size)
            c.drawString(x,height-y-size*.95,s)
        elif op=="box":
            w,h,color=args;c.setFillColor(colors.HexColor(color));c.rect(x,height-y-h,w,h,fill=1,stroke=0)
        else:
            w,h,path=args;c.drawImage(path,x,height-y-h,width=w,height=h,mask="auto")

def draw_ppt(slide,objects):
    for op,x,y,*args in objects:
        if op=="txt":
            s,size,color,width=args
            width=width if width is not None else min(W-x,pdfmetrics.stringWidth(s,"CJK",size)+24)
            size=fit(s,size,width)
            shp=slide.shapes.add_textbox(Pt(x),Pt(y-1),Pt(width+2),Pt(size*1.5))
            tf=shp.text_frame
            tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=Pt(0)
            tf.word_wrap=False
            tf.vertical_anchor=MSO_ANCHOR.TOP
            tf.paragraphs[0].alignment=PP_ALIGN.LEFT
            run=tf.paragraphs[0].add_run();run.text=s
            run.font.name="Microsoft YaHei";run.font.size=Pt(size);run.font.color.rgb=rgb(color)
            ea=OxmlElement("a:ea");ea.set("typeface","Microsoft YaHei");run._r.get_or_add_rPr().append(ea)
        elif op=="box":
            w,h,c=args;shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
            shape.fill.solid();shape.fill.fore_color.rgb=rgb(c);shape.line.fill.background()
            shape.shadow.inherit=False
        else:
            w,h,path=args;slide.shapes.add_picture(path,Pt(x),Pt(y),Pt(w),Pt(h))

def protocol():
    w,h=A4;o=[rect(0,0,w,h,WHITE),rect(0,0,w,9,TEAL),txt(38,28,"ROBOTS  /  评测协议     ·     v2",12,TEAL),
               txt(38,56,"固定最终策略的配对评测",26,INK),txt(39,94,"Cai Renjie    |    Cosmos3-Edge-Policy-DROID × RoboLab    |    2026.11.26",11,MUTED),
               rect(38,122,w-76,1,RULE)]
    def block(y,title,lines):
        o.append(txt(39,y,title,16,TEAL))
        for i,l in enumerate(lines):o.append(txt(39,y+29+i*18,l,11,INK,w-78))
    block(141,"1   固定模型和任务",["原始 Base 为预训练 screening。三个 v2 臂固定 e03c1，参照训练臂为 T′。",
        "A-video 将预测对齐用于视频，A-action 将其加到动作偏好。每组一次训练。",
        "选用既有队列的全部 10 项，每项 12 个 default 回合，均来自 train 候选池，未用于 v2 更新。"])
    labels=["BananaInBowlTask","BananaOnPlateTask","ClampInRightBinTask","CoffeePotInBinTask","FoodPacking1BoxesTask",
            "FoodPacking1CansTask","FruitsOnionToPlateTask","PutBowlOnShelfTopTask","RubiksCubeTask","SauceBottlesCrateTask"]
    o.append(rect(38,230,w-76,92,PALE))
    for i,t in enumerate(labels):
        o.append(txt(50+(i//5)*257,237+(i%5)*16,t,10,MUTED))
    block(341,"2   让起点和预算对应",["每任务一个筛选 seed，run = batch = 0，env_id = 0…11。任务内批 seed 与请求 CRC32 可复算。",
        "各组平移上限 ±0.1 m、yaw 上限 ±60°。一对对应同任务、提示、批次、seed、env_id。",
        "三臂 HDF5 的物体、机器人与相机初始状态在 120 对上相同。逐块核对实际策略版本。"])
    block(439,"3   用仿真终局计分",["成功 = 官方任务放置条件成立，包括夹爪脱离。40/50/60 秒对应 600/750/900 控制步。",
        "到时未成功计 0，途中错抓、掉落记为事件。缺日志、缺起点或作业异常应阻止发布。",
        "每回合一个二元结果。总体成功率为成功次数 / 120，各任务等权，同时公布逐任务计数。"])
    block(537,"4   保留配对关系",["每个起点取两模型成败之差。固定这十项，在每个任务内重采样 12 个整对起点，",
        "共 20,000 次，随机种子 0，报告 2.5% 和 97.5% 分位点及胜/负对数。另给双侧精确",
        "McNemar 检验。区间以当前任务、布局抽样与固定 checkpoint 为条件，未估计跨训练 seed 波动。"])
    block(635,"5   证据与适用范围",["从 Base 的 312 条 default 筛选记录中按固定队列取 120 条，四组按协议键配对。",
        "筛选存档保留 Base 的成败、终止步和 seed。原始 HDF5 已清理，物理初态逐项比对覆盖三组训练策略。",
        "每臂只有一个训练 seed。正式 validation/test、vague 指令与实机需另行评测。"])
    o += [rect(38,750,w-76,1,RULE),txt(39,761,"依据  training/runs/v2_eval_queue.sh · training/v1_eval.py · data/eval_batches.csv",10,MUTED),
          txt(39,778,"原始日志及数表  data/source_records.zip · data/eval_episodes.csv · data/paired.csv",10,MUTED)]
    pdf=Canvas(str(OUT/"评测协议.pdf"),pagesize=A4);pdf.setTitle("固定最终策略的配对评测");draw_pdf(pdf,o,h);pdf.save()

def result_sheet():
    def rows(name):
        with (OUT/"data"/name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    w,h=A4
    obj=[rect(0,0,w,h,WHITE),rect(0,0,w,9,TEAL),txt(36,25,"ROBOTS  /  结果表",12,TEAL),
         txt(36,49,"训练采样与固定模型的实测成绩",23),
         txt(36,87,"同一训练 seed 20261123。固定评测仅用 default，取已筛选 train 池中未参与 v2 更新的 10 项。",10,MUTED,w-72),
         txt(36,105,"正式 test split 此次未测。三组 e03c1 物理初态已核对，Base 原始 HDF5 缺失，仅能复核协议键。",10,MUTED,w-72)]
    xs=[39,150,280,423]
    def line(y,items,heading=False):
        if heading:obj.append(rect(35,y-3,w-70,22,PALE))
        for x,s in zip(xs,items):obj.append(txt(x,y,s,10.5,TEAL if heading else INK))
    line(135,["策略","三轮训练采样","固定模型评测","相对 T′（百分点）"],True)
    for i,r in enumerate(rows("report_table.csv")):
        t="—" if not r["training_success"] else f"{r['training_success']}/288   {float(r['training_pct']):.1f}%"
        score=f"{r['independent_success']}/120   {float(r['independent_pct']):.1f}%"
        if "间接" in r["independent_evidence"]:score+=" *"
        line(163+23*i,[r["model"],t,score,f"{float(r['delta_vs_T_pp']):+.1f}"+(" *" if "间接" in r["independent_evidence"] else "")])
    obj.extend([txt(36,266,"同一起点的成败交换",15,TEAL),
                txt(36,292,"各臂减 T′。区间为固定任务内配对 bootstrap 的 95% 区间，单位是百分点。",10,MUTED)])
    for i,r in enumerate(x for x in rows("paired.csv") if x["reference"]=="T′"):
        obj.append(txt(39,316+i*22,f"{r['model']}     {r['win']} 胜 / {r['loss']} 负 / {r['tie']} 同       差 {float(r['delta_pp']):+.1f}     [{float(r['ci_low_pp']):+.1f}, {float(r['ci_high_pp']):+.1f}]",11))
    obj.append(txt(36,388,"各任务成功数（每格分母为 12）",15,TEAL))
    per=rows("eval_tasks.csv");xx=[39,265,343,411,488]
    obj.append(rect(35,415,w-70,22,PALE))
    for x,s in zip(xx,["任务","原始 Base","T′","A-video","A-action"]):obj.append(txt(x,419,s,10,TEAL))
    for i,t in enumerate(sorted({r["task"] for r in per})):
        vals={r["model"]:r["success"] or "待核" for r in per if r["task"]==t}
        for x,s in zip(xx,[t,vals["Base"],vals["T′"],vals["A-video"],vals["A-action"]]):
            obj.append(txt(x,446+i*20,s,10))
    obj.extend([rect(35,661,w-70,1,RULE),txt(36,677,"各训练臂相对冻结预训练 Base 的配对结果",12,TEAL)])
    for i,r in enumerate(x for x in rows("paired.csv") if x["reference"]=="Base"):
        obj.append(txt(39,700+i*18,
            f"{r['model']}     {r['win']} 胜 / {r['loss']} 负 / {r['tie']} 同       差 {float(r['delta_pp']):+.1f}     [{float(r['ci_low_pp']):+.1f}, {float(r['ci_high_pp']):+.1f}]",
            10))
    obj.extend([txt(36,767,"原始回合    data/source_records.zip     |     逐任务百分比和差值    data/eval_tasks.csv",10,MUTED),
                txt(36,787,"逐回合索引    data/eval_episodes.csv     |     配对区间及精确检验    data/paired.csv",10,MUTED)])
    c=Canvas(str(OUT/"结果表.pdf"),pagesize=A4);c.setTitle("Cai Renjie · 可追溯结果表");draw_pdf(c,obj,h);c.save()

def main():
    ps=Presentation();ps.slide_width=Pt(W);ps.slide_height=Pt(H)
    notes=(OUT/"讲稿与问答.md").read_text()
    notes=re.findall(r"^## 第 .*?\n\n(.*?)(?=\n\n## )",notes,re.S|re.M)
    assert len(notes)==5
    pdf=Canvas(str(OUT/"汇报.pdf"),pagesize=(W,H));pdf.setTitle("Cai Renjie · 仿真、rollout 与评测")
    for obj,speech in zip(deck(),notes):
        slide=ps.slides.add_slide(ps.slide_layouts[6])
        draw_ppt(slide,obj);slide.notes_slide.notes_text_frame.text=speech.strip()
        draw_pdf(pdf,obj,H);pdf.showPage()
    pdf.save();ps.save(OUT/"汇报.pptx");protocol();result_sheet()
if __name__=="__main__":
    main()
