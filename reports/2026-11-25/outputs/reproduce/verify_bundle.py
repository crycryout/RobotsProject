"""Check the portable evidence archive without GPUs, a model, or any H800 paths."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parents[1]
D=HERE/"data"
def csv_rows(p):
    with p.open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def check(value,where):
    if not value:raise ValueError(where)
def main():
    training=csv_rows(D/"training_episodes.csv")
    ep=csv_rows(D/"eval_episodes.csv")
    audit=json.loads((D/"audit.json").read_text())
    check(hashlib.sha256((D/"source_records.zip").read_bytes()).hexdigest()==audit["source_zip_sha256"],"archive sha256")
    with zipfile.ZipFile(D/"initial_states.zip") as states:
        initial={}
        for r in ep:
            if r["model"]=="Base":continue
            name=f"{r['model']}/{r['task']}/env{r['env']}.bin"
            raw_state=states.read(name)
            check(hashlib.sha256(raw_state).hexdigest()==r["initial_state_sha256"],name)
            key=(r["task"],r["env"])
            check(initial.setdefault(key,raw_state)==raw_state,key)
        check(len(initial)==120 and len(states.namelist())==360,"initial state coverage")
    raw_eval={}
    with zipfile.ZipFile(D/"source_records.zip") as z:
        check(z.testzip() is None,"archive CRC")
        for r in ep:
            if r["model"]=="Base":
                file="screening/"+Path(r["source"]).name
                body=z.read(file)
                base=pq.ParquetFile(io.BytesIO(body)).read(use_threads=False).to_pylist()
                actual,=[rec for rec in base if rec["env"]==int(r["env"])]
                check((actual["task"],actual["level"],actual["seed"],actual["policy_version"])==
                      (r["task"],r["level"],int(r["seed"]),"base") and type(actual["success"]) is bool,file)
            else:
                suffix=PurePosixPath(r["source"]).parts[-6:]
                body=z.read("scratch/"+"/".join(suffix))
                actual=json.loads(body)
                check(actual["task"]==r["task"] and actual["env_id"]==int(r["env"]),r["source"])
            check(hashlib.sha256(body).hexdigest()==r["source_sha256"],r["source"])
            check(int(actual["success"])==int(r["success"]),r["source"])
            key=(r["task"],r["level"],r["batch"],r["seed"],int(r["env"]))
            slot=raw_eval.setdefault(r["model"],{})
            check(key not in slot,key)
            slot[key]=int(actual["success"])
        files={}
        for r in training:
            name="repo/"+r["source"]
            if name not in files:files[name]=pq.ParquetFile(io.BytesIO(z.read(name))).read(use_threads=False).to_pylist()
            rec=files[name][int(r["row_zero_based"])]
            check((rec["task"],rec["level"],rec["seed"],rec["env"],int(rec["success"]),rec["policy_version"])==
                  (r["task"],r["level"],int(r["seed"]),int(r["env"]),int(r["success"]),r["policy_version"]),name)
    rates=csv_rows(D/"results.csv")
    if "Base" in raw_eval:
        checklist=json.loads((D/"screening_crosscheck.json").read_text())
        reference=D/"screening_base_20261123.zip"
        check(hashlib.sha256(reference.read_bytes()).hexdigest()==checklist["snapshot_sha256"],reference)
        with zipfile.ZipFile(reference) as screening:
            check(screening.testzip() is None,reference)
            original=pq.ParquetFile(io.BytesIO(screening.read("episodes_default.parquet"))).read(use_threads=False).to_pylist()
        use=set(checklist["selected_tasks"])
        base={(r["task"],r["level"],"0",str(r["seed"]),r["env"]):int(r["success"]) for r in original if r["task"] in use}
        check(len(original)==312 and len(base)==120 and base==raw_eval["Base"],"independent merged screening")
    for rm in rates:
        if rm["status"]!="verified":continue
        if rm["dataset"]=="training_rollout":s=[int(r["success"]) for r in training if r["model"]==rm["model"]]
        else:s=list(raw_eval[rm["model"]].values())
        check((sum(s),len(s))==(int(rm["success"]),int(rm["episodes"])),rm)
    for row in csv_rows(D/"paired.csv"):
        a,b=raw_eval[row["model"]],raw_eval[row["reference"]]
        check(set(a)==set(b),row)
        ds=np.array([a[k]-b[k] for k in sorted(a)]).reshape(10,12)
        wins,losses=int((ds>0).sum()),int((ds<0).sum())
        check((wins,losses,int((ds==0).sum()),ds.size)==tuple(int(row[k]) for k in ("win","loss","tie","pairs")),row)
        check(np.isclose(100*ds.mean(),float(row["delta_pp"])),row)
        discordant=wins+losses
        p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(wins,losses)+1))/2**discordant)
        check(np.isclose(p,float(row["exact_mcnemar_p"])),row)
        rng=np.random.default_rng(0)
        draws=sum(rng.choice(group,(20000,12)).sum(axis=1) for group in ds)
        bounds=np.percentile(draws*100/ds.size,[2.5,97.5])
        check(np.allclose(bounds,[float(row["ci_low_pp"]),float(row["ci_high_pp"])]),row)
    from training_time_steps import main as verify_training_steps
    verify_training_steps(check_only=True)
    print(f"OK: {len(training)} training rows, {len(ep)} evaluation rows, {len(raw_eval['T′'])} aligned keys; raw counts and paired intervals agree.")
if __name__=="__main__":main()
