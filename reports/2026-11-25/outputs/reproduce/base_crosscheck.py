"""Compare the historical v1 aggregate, raw v1 outcomes and recovered screening results."""
import argparse
import csv
from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path
import zipfile

import h5py

from audit import N, REPO, layout_hash, tasks_from_queue, trace_version, require


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--scratch",type=Path,default=Path("/var/tmp/vis_rl"))
    ap.add_argument("--data",type=Path,required=True)
    args=ap.parse_args()
    out=args.data
    with (out/"eval_episodes.csv").open(encoding="utf-8-sig") as f:
        episodes=list(csv.DictReader(f))
    fixed={(r["task"],int(r["env"])):r for r in episodes if r["model"]=="T′"}
    direct={(r["task"],int(r["env"])):r for r in episodes if r["model"]=="Base"}
    rows=[];files=[];versions=set()
    for t in tasks_from_queue():
        batch=args.scratch/"eval_eval_untrained_e02c2/model1"/f"{t}_default_b0"
        sim=batch/"rollouts"/t
        seed=json.loads((sim/"env_cfg.json").read_text())["seed"]
        with h5py.File(sim/"run_0.hdf5") as traj:
            for env in range(N):
                p=sim/f"log_0_env{env}.json"
                log=json.loads(p.read_text())
                current=fixed[t,env]
                ver=trace_version(batch/"capture"/f"{t}_run0_env{env}_step0000_nft.pt")
                require(log["env_id"]==env and log["run"]==0 and log["task"]==t and log["instruction_fields_used"]==["default"],p)
                require(seed==int(current["seed"]) and layout_hash(traj[f"data/demo_{env}/initial_state"])==current["initial_state_sha256"],p)
                require(type(log["success"]) is bool and log["success"]==bool(traj[f"data/demo_{env}"].attrs["success"]),p)
                versions.add(ver);files.append(p)
                rows.append(dict(task=t,env=env,success=int(log["success"]),seed=seed,version=ver,
                                 source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    require(len(versions)==1 and len(rows)==len(fixed)==120,versions)
    note=REPO/"train_v2.md"
    line,= [line for line in note.read_text().splitlines() if "e02c2" in line and "Δsuccess" in line]
    report=re.search(r"Δsuccess\s+([−+\-0-9.]+).*?n = (\d+)",line)
    delta=Decimal(report[1].replace("−","-")); n=int(report[2]); trained=sum(r["success"] for r in rows)
    require(n==len(rows),line)
    candidates=[count for count in range(n+1) if round(Decimal(trained-count)/n,3)==delta]
    require(len(candidates)==1,"Ambiguous rounded aggregate")
    evidence=dict(source_doc="train_v2.md §1",source_text=line,older_checkpoint="v1 e02c2",v1_policy_version=versions.pop(),
                  v1_success=trained,n=n,published_v1_minus_base=float(delta),implied_base_success=candidates[0],
                  matched_initial_layouts=len(rows),evidence_level="secondary inference; screening per-episode outcomes unobserved")
    if direct:
        require(set(direct)==set(fixed), "Historical and screening cohorts differ")
        require(all(int(direct[r["task"],r["env"]]["seed"])==r["seed"] for r in rows), "Historical seed mismatch")
        actual=sum(int(r["success"]) for r in direct.values())
        require(actual==candidates[0], "Published aggregate does not match the original screening")
        evidence.update(direct_base_success=actual, matched_screening_keys=len(direct),
                        v1_only_wins=sum(r["success"]>int(direct[r["task"],r["env"]]["success"]) for r in rows),
                        base_only_wins=sum(r["success"]<int(direct[r["task"],r["env"]]["success"]) for r in rows),
                        evidence_level="Direct frozen-base episode outcomes agree with the separate historical aggregate.",
                        matched_initial_layouts_scope="v1 e02c2 versus v2 T-prime HDF5; screening pairs are verified by keys.")
    (out/"base_crosscheck.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    with (out/"v1_reference_episodes.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    with zipfile.ZipFile(out/"v1_reference_records.zip","x",compression=zipfile.ZIP_DEFLATED) as z:
        z.write(note,"train_v2.md")
        for p in files:z.write(p,str(p.relative_to(args.scratch)))
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
