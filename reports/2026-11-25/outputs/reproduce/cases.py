"""Select reproducible examples and recover real sensor keyframes from fixed-checkpoint captures."""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1]
PILOT = Path("/home/robot_share/task_sweep/dream_pilot_20261121/rollouts")
FONT = os.environ.get("CJK_FONT", "/home/renjie/qwen_pcie/fonts/NotoSansSC-400.ttf")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick(records):
    for task in sorted({r["task"] for r in records}):
        group = sorted((r for r in records if r["task"] == task), key=lambda r: (r["run"],r["env_id"]))
        if {r["success"] for r in group} == {True, False}:
            return [next(r for r in group if r["success"] == good) for good in (True,False)]
    raise ValueError("No task contains both outcomes")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=[]
    source=[]
    for p in sorted(PILOT.glob("*/log_*_env*.json")):
        row=json.loads(p.read_text())
        row["path"]=p
        source.append(row)
    for row in pick(source):
        p=row["path"]
        media,=p.parent.glob(f"*_{row['run']}_env{row['env_id']}.mp4")
        tag="success" if row["success"] else "timeout"
        name=f"pilot_{tag}.mp4"
        shutil.copyfile(media,args.out/name)
        shutil.copyfile(p,args.out/f"pilot_{tag}.json")
        meta.append(dict(dataset="2026-11-21 pilot",file=name,task=row["task"],env=row["env_id"],run=row["run"],
                         success=int(row["success"]),final_step=row["final_step"],source=str(media),raw_log=str(p),
                         source_sha256=digest(media),output_sha256=digest(args.out/name),
                         video_type="原始连续传感器录像",encoded_fps=15,distinct_frames_per_playback_second=15,
                         frame_selection="原始 RoboLab MP4，字节与源文件一致"))
    source=[]
    with (OUT/"data/eval_episodes.csv").open(encoding="utf-8-sig") as f:
        for record in csv.DictReader(f):
            if record["model"]=="T′":
                p=Path(record["source"])
                row=json.loads(p.read_text())
                row["path"]=p
                source.append(row)
    normal=ImageFont.truetype(FONT,23)
    small=ImageFont.truetype(FONT,17)
    for row in pick(source):
        p=row["path"]
        cap=p.parents[2]/"capture"
        base=f"{row['task']}_run{row['run']}_env{row['env_id']}_step"
        inputs=[]
        for step in range(0,row["final_step"],32):
            request=cap/f"{base}{step:04d}_request.npz"
            with np.load(request,allow_pickle=False) as z:
                inputs.append((step,Image.fromarray(z["image"])))
        terminal=cap/row["v1_progress"]["terminal_frame"]["file"]
        inputs.append((row["final_step"],Image.fromarray(np.load(terminal,allow_pickle=False))))
        tag="success" if row["success"] else "timeout"
        label="成功" if row["success"] else "超时"
        name=f"eval_T_{tag}.mp4"
        # Two sampled frames per second. Preserve original timestamps as control-step labels.
        cmd=["ffmpeg","-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s","640x632",
             "-r","2","-i","pipe:0","-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-r","24",
             "-movflags","+faststart","-n",str(args.out/name)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
            for step,im in inputs:
                frame=Image.new("RGB",(640,632),"#102738")
                frame.paste(im,(0,52))
                draw=ImageDraw.Draw(frame)
                draw.text((16,9),f"T′  ·  香蕉放盘    env {row['env_id']}    终局{label}",font=normal,fill="white")
                draw.text((16,598),f"真实观测关键帧    控制步 {step:03d} / {row['final_step']}    仿真 {step/15:.1f} 秒",font=small,fill="white")
                encoder.stdin.write(frame.tobytes())
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed")
        meta.append(dict(dataset="e03c1 T′ keyframes",file=name,task=row["task"],env=row["env_id"],run=row["run"],
                         success=int(row["success"]),final_step=row["final_step"],source=str(cap),raw_log=str(p),
                         source_sha256=digest(p),output_sha256=digest(args.out/name),
                         video_type="采样帧回放",encoded_fps=24,distinct_frames_per_playback_second=2,
                         frame_selection="从控制步 0 起每 32 步取真实请求帧，另加终帧"))
    with (args.out/"manifest.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=meta[0]);writer.writeheader();writer.writerows(meta)
    for r in meta:
        print(r["dataset"],r["file"],r["task"],r["env"],r["final_step"])
if __name__=="__main__":
    main()
