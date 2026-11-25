"""Rebuild the course results from captured episodes. All metrics use the fixed v2 protocol."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest
from report_scope import annotate

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from training.v1_reward import episode_chunks
from training.v1_eval import trace_version
from training.v1_run import stable_seed

ARMS = {"T′": ("v2_T_01", "eval_eval_untrained_ep3"),
        "A-video": ("v2_AV_01", "eval_v2_AV_01_eval_untrained_ep3"),
        "A-action": ("v2_AA_01", "eval_v2_AA_01_eval_untrained_ep3")}
BASE = "Base"
N = 12
SEED = 20261123


def load(path):
    return json.loads(path.read_text())


def jsonlines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(str(message))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def tasks_from_queue():
    script = (REPO / "training/runs/v2_eval_queue.sh").read_text()
    tasks = re.search(r'^tasks="([^"]+)"', script, re.M)[1].split()
    require(len(tasks) == len(set(tasks)) == 10, "Unexpected task queue")
    return sorted(tasks)


def layout_bytes(group):
    packed = bytearray()
    def member(name, obj):
        if isinstance(obj, h5py.Dataset):
            array = obj[()]
            packed.extend(f"{name}|{array.dtype}|{array.shape}|".encode())
            packed.extend(array.tobytes())
    group.visititems(member)
    return bytes(packed)


def layout_hash(group):
    return hashlib.sha256(layout_bytes(group)).hexdigest()


def read_training():
    episodes, epochs, per_task, versions, sources, configs = [], [], [], {}, set(), {}
    for name, (run, _) in ARMS.items():
        root = REPO / "training/runs" / run
        state = load(root / "state.json")
        status = load(root / "status.json")
        require(status["phase"] == "done" and state["checkpoint"].endswith("/e03c1"), root)
        versions[name] = state["version"]
        history = jsonlines(root / "history.jsonl")
        require(len(history) == 6 and history[-1]["version"] == versions[name], root)
        # The published GPU UUID lines are malformed after redaction. Parse a copy in memory
        # for the other fields, keeping the original bytes in the evidence archive.
        raw = (root / "config.json").read_text()
        preview = re.sub(r'"gpus":\s*\[[^\]]*\]', '"gpus": []', raw, count=1)
        cfg = json.loads(preview)
        configs[name] = {k: cfg[k] for k in ("pairs","num_envs","seed","init_pose_noise","arm","gamma","passes",
                                              "lora_rank","learning_rate","beta","real16","alignment_bound","max_epochs")}
        for q in ("config.json","state.json","status.json","history.jsonl"):
            sources.add(root / q)
        previous = "base"
        arm_rows = []
        for item in history:
            c = item["collection"]
            ep_file = root / "counters" / f"episodes_{c}.parquet"
            chunk_file = root / "counters" / f"chunks_{c}.parquet"
            sources.update([ep_file, chunk_file])
            rows = pq.read_table(ep_file).to_pylist()
            require(len(rows) == item["episodes"] == 48 and sum(r["success"] for r in rows) == item["successes"], ep_file)
            require(sum(r["steps"] for r in rows) == item["env_steps"], ep_file)
            chunks = pq.read_table(chunk_file, columns=["batch","env","start","end","executed","terminal","success",
                                                        "p_start","p_end","reward","p_action","p_action_used","p_video",
                                                        "align_gain"]).to_pylist()
            require(len(chunks) == item["chunks"], chunk_file)
            by_env = defaultdict(list)
            gains = []
            for c_row in chunks:
                by_env[c_row["batch"],c_row["env"]].append(c_row)
                gain = c_row["align_gain"]
                valid = gain is not None and math.isfinite(gain)
                z = float(np.clip(gain/cfg["alignment_bound"], -1, 1)) if valid else 0.
                if valid:
                    require(c_row["executed"] >= 16, chunk_file)
                    gains.append(gain)
                expected_video = 0.5 + 0.5*z if name == "A-video" else 0.5
                expected_action = np.clip(c_row["p_action"] + 0.1*z, 0, 1) if name == "A-action" else c_row["p_action"]
                require(np.isclose(expected_video,c_row["p_video"]) and np.isclose(expected_action,c_row["p_action_used"]), chunk_file)
                require(c_row["reward"] == c_row["p_end"] - c_row["p_start"] + c_row["success"], chunk_file)
            require(np.isclose(len(gains)/len(chunks),item["align_coverage"]) and np.isclose(np.mean(gains),item["align_mean"]), chunk_file)
            for i, r in enumerate(rows):
                level, task, epoch = r["level"], r["task"], r["epoch"]
                seed = stable_seed(SEED,"track",task,"default") if level == "default" else stable_seed(SEED,"fresh",task,level,epoch)
                require(r["seed"] == seed and r["policy_version"] == previous and r["arm"] == cfg["arm"], ep_file)
                seq = sorted(by_env[r["batch"],r["env"]], key=lambda c:c["start"])
                require([c["start"] for c in seq] == list(range(0,r["steps"],32)), ep_file)
                require(sum(c["executed"] for c in seq) == r["steps"] and sum(c["terminal"] for c in seq) == 1
                        and seq[-1]["terminal"] and seq[-1]["success"] == int(r["success"]), ep_file)
                arm_rows.append(dict(model=name,epoch=epoch,collection=r["collection"],task=task,level=level,
                                     seed=seed,env=r["env"],success=int(r["success"]),steps=r["steps"],policy_version=previous,
                                     source=str(ep_file.relative_to(REPO)),row_zero_based=i))
            previous = item["version"]
        require(sum(r["steps"] for r in arm_rows)==state["env_steps"],root)
        for epoch in (1,2,3):
            rs = [r for r in arm_rows if r["epoch"]==epoch]
            keys = {(r["task"],r["level"],r["env"]) for r in rs}
            require(len(rs)==len(keys)==96 and keys=={(t,l,e) for t,l in cfg["pairs"] for e in range(N)}, epoch)
            epochs.append(dict(model=name,epoch=epoch,success=sum(r["success"] for r in rs),episodes=len(rs)))
        for t in sorted({r["task"] for r in arm_rows}):
            for level in sorted({r["level"] for r in arm_rows if r["task"]==t}):
                rs=[r for r in arm_rows if r["task"]==t and r["level"]==level]
                per_task.append(dict(model=name,task=t,level=level,success=sum(r["success"] for r in rs),episodes=len(rs)))
        episodes.extend(arm_rows)
    require(len({tuple(map(tuple, x["pairs"])) for x in configs.values()})==1,configs)
    return episodes, epochs, per_task, versions, sources, configs


def read_eval(scratch, tasks, versions, sources):
    rows, batches, files, states = [], [], [], {}
    split = {rec["task"]:(s,rec) for s,rs in load(REPO/"training/task_split.json")["splits"].items() for rec in rs}
    sources.update([REPO/"training/task_split.json",REPO/"training/runs/v2_eval_queue.sh"])
    for model, (_, dirname) in ARMS.items():
        root = scratch/dirname/"model1"
        require({p.name for p in root.iterdir() if p.is_dir()} == {f"{t}_default_b0" for t in tasks},root)
        for t in tasks:
            path=root/f"{t}_default_b0"
            sims=path/"rollouts"/t
            cfgfile=sims/"env_cfg.json"
            posefile=sims/"init_pose_groups.json"
            epfile=path/"rollouts/episode_results.jsonl"
            h5file=sims/"run_0.hdf5"
            config=load(cfgfile)
            pose=load(posefile)
            seed=config["seed"]
            expected_seed=stable_seed(SEED,"track",t,"default")
            terms=config["terminations"]
            require(config["_task_name"]==t and seed==expected_seed and config["num_envs"]==N and config["policy"]=="cosmos3",cfgfile)
            require(config["events"]["randomize_init_pose"]["params"]["noise_m"]==pose["noise_m"]==0.1
                    and abs(pose["yaw_range_deg"]-60)<1e-7 and len(terms)==2 and terms["time_out"]["time_out"]
                    and terms["success"]["params"]["require_gripper_detached"],cfgfile)
            dt=config["sim"]["dt"]*config["decimation"]
            horizon=round(config["episode_length_s"]/dt)
            require(np.isclose(dt,1/15), cfgfile)
            preserved=jsonlines(epfile)
            official={r["env_id"]:r for r in preserved}
            logs=sorted(sims.glob("log_*_env*.json"))
            require(len(logs)==len(preserved)==len(official)==N and set(official)==set(range(N)),epfile)
            files.extend([cfgfile,posefile,epfile,*logs])
            batches.append(dict(model=model,task=t,split=split[t][0],scene=split[t][1]["scene"],seed=seed,
                                num_envs=N,noise_m=pose["noise_m"],yaw_deg=60,horizon_control_steps=horizon,
                                official_success=terms["success"]["func"],config=str(cfgfile),config_sha256=sha(cfgfile),
                                trajectory=str(h5file),trajectory_sha256=sha(h5file)))
            with h5py.File(h5file) as f:
                for logfile in logs:
                    log=load(logfile)
                    env=log["env_id"]
                    orig=official[env]
                    length=log["final_step"]
                    good=log["success"]
                    demo=f[f"data/demo_{env}"]
                    require(log["run"]==orig["run"]==0 and log["task"]==orig["task_name"]==t
                            and log["instruction_fields_used"]==orig["instruction_fields_used"]==["default"]
                            and orig["instruction"]==split[t][1]["prompts"]["default"], logfile)
                    require(type(good) is bool and good==orig["success"]==bool(demo.attrs["success"])
                            and length==orig["episode_step"]==len(demo["actions"]) and np.isclose(log["dt"],dt)
                            and 0<length<=horizon and (good or length==horizon), logfile)
                    chunks=episode_chunks(log)
                    packed=layout_bytes(demo["initial_state"])
                    initial=hashlib.sha256(packed).hexdigest()
                    states[f"{model}/{t}/env{env}.bin"]=packed
                    commands=demo["actions"][()]
                    seed0=None
                    for ch in chunks:
                        capture=path/"capture"
                        trace=capture/f"{ch.key}_nft.pt"
                        require(trace_version(trace)==versions[model],trace)
                        req=capture/f"{ch.key}_request.npz"
                        with np.load(req,allow_pickle=False) as d:
                            cmd=d["command"]
                            action=d["action"]
                            request_seed=int(d["seed"])
                            require(request_seed==(zlib.crc32(f"{seed}:{ch.key}".encode()) & 0xffffffff),req)
                            require(action.shape==cmd.shape==(32,8) and np.array_equal(action[:,:7],cmd[:,:7])
                                    and np.array_equal(action[:,-1]>0.5,cmd[:,-1])
                                    and np.array_equal(cmd[:ch.executed],commands[ch.start:ch.end]),req)
                            if ch.start==0:
                                seed0=request_seed
                                require(d["image"].shape==(540,640,3) and d["joint"].shape==(7,) and d["gripper"].shape==(1,)
                                        and str(d["prompt"])==orig["instruction"],req)
                    terminal=log["v1_progress"]["terminal_frame"]
                    require(terminal["executed"]==chunks[-1].executed and (capture/terminal["file"]).is_file(),logfile)
                    rows.append(dict(model=model,task=t,level="default",batch=0,seed=seed,env=env,success=int(good),
                                     steps=length,termination="success" if good else "timeout",chunks=len(chunks),
                                     initial_state_sha256=initial,request_seed_step0=seed0,policy_version=versions[model],
                                     source=str(logfile),source_sha256=sha(logfile)))
    return rows,batches,files,states


def screening_files(directory, tasks):
    """Locate one original episode table per queue task in a screening archive."""
    directory = directory.expanduser().resolve(strict=True)
    expected = {f"episodes_screen_{t}.parquet": t for t in tasks}
    found = defaultdict(list)
    for path in directory.rglob("episodes_screen_*.parquet"):
        if path.name in expected and path.is_file():
            found[expected[path.name]].append(path)
    for task in tasks:
        matches = found[task]
        require(len(matches) == 1,
                f"Expected one episodes_screen_{task}.parquet under {directory}, found {len(matches)}: {matches}")
    return {task: found[task][0] for task in tasks}


def screening(files, rows, batches):
    horizons = {batch["task"]: batch["horizon_control_steps"] for batch in batches}
    for t, src in files.items():
        rs=pq.read_table(src).to_pylist()
        require(len(rs)==N and all(type(r["env"]) is int for r in rs) and {r["env"] for r in rs}==set(range(N)),src)
        source_hash = sha(src)
        for r in rs:
            require(r["task"]==t and r["level"]=="default" and r["seed"]==stable_seed(SEED,"track",t,"default")
                    and r["policy_version"]=="base" and type(r["success"]) is bool,src)
            require(r.get("collection", "screen") == "screen" and r.get("epoch", 0) == 0
                    and r.get("batch", f"screen_{t}") == f"screen_{t}", src)
            horizon = horizons[t]
            require(type(r["steps"]) is int and 0 < r["steps"] <= horizon
                    and (r["success"] or r["steps"] == horizon)
                    and (r["final_p"] == 1.0) == r["success"], src)
            rows.append(dict(model=BASE,task=t,level="default",batch=0,seed=r["seed"],env=r["env"],success=int(r["success"]),
                             steps=r["steps"],termination="success" if r["success"] else "timeout",chunks="",
                             initial_state_sha256="",request_seed_step0="",policy_version="base",source=str(src),
                             source_sha256=source_hash))


def results(rows, training, tasks):
    by_model=defaultdict(dict)
    for r in rows:
        key=(r["task"],r["level"],r["batch"],r["seed"],r["env"])
        require(key not in by_model[r["model"]],key)
        by_model[r["model"]][key]=r
    keys=sorted(by_model["T′"])
    for model, group in by_model.items():
        require(set(group)==set(keys) and len(keys)==120,model)
        if model!=BASE:
            require(all(group[k]["initial_state_sha256"]==by_model["T′"][k]["initial_state_sha256"]
                        and group[k]["request_seed_step0"]==by_model["T′"][k]["request_seed_step0"] for k in keys),model)
    totals, detail, comparisons = [], [], []
    for role, seq in (("training_rollout",training),("untrained_task_eval",rows)):
        for model in (BASE,*ARMS):
            sample=[r for r in seq if r["model"]==model]
            s=sum(r["success"] for r in sample)
            def diff(ref):
                rr=[r["success"] for r in seq if r["model"]==ref]
                return round(100*(s/len(sample)-sum(rr)/len(rr)),6) if rr and sample else ""
            totals.append(dict(dataset=role,model=model,success=s if sample else "",episodes=len(sample) if sample else "",
                               success_pct=round(100*s/len(sample),6) if sample else "",delta_vs_T_pp=diff("T′"),
                               delta_vs_base_pp=diff(BASE),status="verified" if sample else
                               ("not_applicable" if role=="training_rollout" else "screening_rows_unavailable")))
    for t in tasks:
        for model in (BASE,*ARMS):
            rs=[r["success"] for r in rows if r["task"]==t and r["model"]==model]
            t_count=sum(by_model["T′"][k]["success"] for k in keys if k[0]==t)
            detail.append(dict(task=t,model=model,success=sum(rs) if rs else "",episodes=len(rs) if rs else "",
                               success_pct=round(100*np.mean(rs),6) if rs else "",
                               delta_vs_T_pp=round(100*(sum(rs)-t_count)/N,6) if rs else "",
                               delta_vs_base_pp=round(100*(sum(rs)-sum(by_model[BASE][k]["success"] for k in keys if k[0]==t))/N,6)
                               if rs and BASE in by_model else ""))
    for ref in ("T′",BASE):
        if ref not in by_model:
            continue
        for model in ARMS:
            if model==ref:
                continue
            # Conditional on these ten tasks: resample whole matched layout differences inside each task.
            diff=np.array([by_model[model][k]["success"]-by_model[ref][k]["success"] for k in keys]).reshape(len(tasks),N)
            rng=np.random.default_rng(0)
            resamples=np.zeros(20000)
            for task_d in diff:
                resamples+=rng.choice(task_d,size=(len(resamples),N)).sum(axis=1)
            lo,hi=np.percentile(100*resamples/diff.size,[2.5,97.5])
            w=int((diff==1).sum()); loss=int((diff==-1).sum())
            p=float(binomtest(w,w+loss,p=.5).pvalue) if w+loss else 1.
            comparisons.append(dict(model=model,reference=ref,pairs=diff.size,win=w,loss=loss,tie=diff.size-w-loss,
                                    delta_pp=round(100*diff.mean(),6),ci_low_pp=round(lo,6),ci_high_pp=round(hi,6),
                                    exact_mcnemar_p=p))
    return totals,detail,comparisons


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch",type=Path,default=Path("/var/tmp/vis_rl"))
    ap.add_argument("--screening-counters",type=Path,help="Directory holding the original per-task screening Parquets, directly or in subdirectories")
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    tasks=tasks_from_queue()
    base_files=screening_files(args.screening_counters,tasks) if args.screening_counters is not None else {}
    train,epochs,per_train,versions,sources,configs=read_training()
    require(not {r["task"] for r in train}.intersection(tasks),tasks)
    rows,batches,raw,states=read_eval(args.scratch,tasks,versions,sources)
    screening(base_files,rows,batches)
    totals,detail,comparisons=results(rows,train,tasks)
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    require(not (out/"results.csv").exists(),"Use a new analysis output directory")
    for name,data in [("results",totals),("training_epochs",epochs),("training_tasks",per_train),
                      ("training_episodes",train),("eval_tasks",detail),("eval_episodes",rows),
                      ("eval_batches",batches),("paired",comparisons)]:
        table(out/f"{name}.csv",[annotate(name,row) for row in data])
    with zipfile.ZipFile(out/"source_records.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(sources):
            archive.write(f,"repo/"+str(f.relative_to(REPO)))
        for f in raw:
            archive.write(f,"scratch/"+str(f.relative_to(args.scratch)))
        for f in base_files.values():
            archive.write(f,"screening/"+f.name)
    with zipfile.ZipFile(out/"initial_states.zip","x") as archive:
        for name, payload in sorted(states.items()):
            entry=zipfile.ZipInfo(name)
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,payload)
    facts=dict(versions=versions,configs=configs,fixed_layouts=120,tasks=tasks,epochs=epochs,totals=totals,paired=comparisons,
               checked_eval_chunks=sum(int(r["chunks"] or 0) for r in rows),source_zip_sha256=sha(out/"source_records.zip"))
    (out/"audit.json").write_text(json.dumps(facts,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:facts[k] for k in ("checked_eval_chunks","fixed_layouts","versions","totals","paired")},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
