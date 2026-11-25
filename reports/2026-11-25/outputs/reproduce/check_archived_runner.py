"""Exercise the repaired evaluator's scoring path on all three real archives in temporary outputs."""
import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path

from audit import ARMS, screening_files, tasks_from_queue
from training import v1_eval as ev

OUT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=OUT/"data")
    parser.add_argument("--screening-counters", type=Path)
    args=parser.parse_args()
    tasks=tasks_from_queue()
    base_files=screening_files(args.screening_counters,tasks) if args.screening_counters else {}
    with (args.data/"eval_batches.csv").open(encoding="utf-8-sig") as f:
        by_key={(r["model"],r["task"]):r for r in csv.DictReader(f)}
    facts=json.loads((args.data/"audit.json").read_text())
    rows={}
    for model,(run,_) in ARMS.items():
        model_path=f"training/runs/{run}/checkpoints/e03c1"
        for task in tasks:
            source=by_key[model,task]
            cfg_file=Path(source["config"])
            batch=dict(task=task,level="default",index=0,seed=int(source["seed"]))
            r=ev.episode_rows(cfg_file.parents[2],batch,model_path,facts["versions"][model])
            ev.check_batch(r,batch,12,model_path)
            ev.check_runtime(json.loads(cfg_file.read_text()),batch,{"num_envs":12,"init_pose_noise":0.1},r)
            rows[model,task]=(r,batch,model_path)
    for comparison in facts["paired"]:
        ref, model=comparison["reference"],comparison["model"]
        if ref=="Base" and not base_files:
            continue
        with tempfile.TemporaryDirectory() as rootname:
            root=Path(rootname)
            plans=[rows[model,task][1] for task in tasks]
            names=[]; versions=[]
            run=root/"baseline"
            for i,tag in enumerate((ref,model)):
                if tag=="Base":
                    counters=run/"counters"; counters.mkdir(parents=True)
                    for source in base_files.values():
                        shutil.copyfile(source,counters/source.name)
                    names.append("screening"); versions.append("screening")
                    continue
                slot=root/f"model{i}";slot.mkdir()
                for task in tasks:
                    r,b,path=rows[tag,task]
                    ev.write_json(slot/f"{task}_default_b0.json",{"rows":r,"version":facts["versions"][tag]})
                names.append(rows[tag,tasks[0]][2]); versions.append(facts["versions"][tag])
            plan={"models":names,"tasks":tasks, "level":"default","batches":plans,"run":str(run),
                  "baseline":"screening" if ref=="Base" else "run","init_pose_noise":0.1,"num_envs":12}
            ev.write_json(root/"plan.json",plan)
            ev.summarize(root,versions)
            summary=ev.read_json(root/"summary.json")
            expected=comparison["delta_pp"]
            if summary["n_pairs"]!=120 or abs(100*summary["paired"]["success"][0]-expected)>1e-8:
                raise ValueError(f"Repaired evaluator disagrees for {model}")
            print(f"OK: official scoring readback {model} versus {ref}, 120 pairs, {expected} percentage points.")
    print("Checked real seed, noise, official horizon, model/episode identity in all 30 batches.")
if __name__=="__main__":main()
