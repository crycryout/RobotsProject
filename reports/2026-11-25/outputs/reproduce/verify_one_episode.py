"""Check a new model-driven episode from the policy response to the simulator's terminal state."""
import argparse
import csv
import json
import math
import re
import subprocess
import zlib
from pathlib import Path

import h5py
import numpy as np

from check_episode_model import REPO, REV


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Verify recorded outputs without writing")
    args = parser.parse_args()
    run = args.run
    out = run / "rollout"
    task = "BananaInBowlTask"
    folder = out / task
    cfg = load(folder / "env_cfg.json")
    client = load(out / "client_config.json")
    model = load(run / "model.json")
    assert (model["repo_id"], model["revision"]) == (REPO, REV)
    assert len(model["files"]) == 34
    assert client["task"] == [task] and client["seed"] == cfg["seed"] == 2105981890
    assert client["num_envs"] == cfg["scene"]["num_envs"] == 1 and client["num_runs"] == 1
    assert client["instruction_type"] == ["default"] and client["video_mode"] == "sensor"
    assert np.allclose(client["init_pose_noise"], [0.1, 60])
    record, = [json.loads(line) for line in (out / "episode_results.jsonl").read_text().splitlines() if line.strip()]
    log = load(folder / "log_0_env0.json")
    steps = record["episode_step"]
    good = record["success"]
    assert type(good) is bool and good == log["success"] and steps == log["final_step"]
    assert record["env_id"] == record["run"] == 0 and record["task_name"] == log["task"] == task
    assert record["instruction_fields_used"] == log["instruction_fields_used"] == ["default"]
    dt = cfg["sim"]["dt"] * cfg["decimation"]
    horizon = round(cfg["episode_length_s"] / dt)
    assert np.isclose(dt,record["dt"]) and np.isclose(dt,1/15) and horizon == 750
    assert 0 < steps <= horizon and (good or steps == horizon)
    with h5py.File(folder / "run_0.hdf5", "r") as stream:
        demo = stream["data/demo_0"]
        commands = demo["actions"][()]
        assert len(demo["initial_state"]) > 0 and bool(demo.attrs["success"]) == good
    assert commands.shape == (steps, 8) and np.isfinite(commands).all()
    chunks = sorted((out / "policy_chunks").glob("*.npz"))
    assert len(chunks) == math.ceil(steps/32)
    action_rows = []
    for start, source in zip(range(0, steps, 32), chunks):
        key = f"{task}_run0_env0_step{start:04d}"
        assert source.stem == key
        with np.load(source, allow_pickle=False) as capture:
            seed = int(capture["seed"])
            assert seed == (zlib.crc32(f"{client['seed']}:{key}".encode()) & 0xffffffff)
            assert str(capture["prompt"]) == record["instruction"] and capture["image"].shape == (540,640,3)
            raw = capture["action"]
            assert raw.shape == (32,8) and np.isfinite(raw).all()
            expected = raw.astype(commands.dtype, copy=True)
            expected[:,7] = raw[:,7] > .5
            used = min(steps-start, 32)
            assert np.array_equal(expected[:used], commands[start:start+used])
            for offset in range(used):
                action_rows.append([start+offset+1,start,offset,seed,*commands[start+offset].tolist()])
    assert len(action_rows) == steps
    video, = folder.glob("*.mp4")
    decoded = subprocess.run(["ffmpeg","-nostdin","-hide_banner","-xerror","-nostats","-progress","pipe:1",
                              "-i",str(video),"-map","0:v:0","-f","null","-"],capture_output=True,text=True,check=True)
    frames = int(re.findall(r"^frame=(\d+)$", decoded.stdout, re.M)[-1])
    fps = float(re.search(r"([0-9.]+) fps", decoded.stderr)[1])
    # Upstream skips writing a sensor frame for an env once it becomes frozen on its terminal step.
    assert fps == 15 and frames in (steps-1, steps)
    summary = dict(purpose="单独的冻结 Base 端到端演示，排除在四组 480 条评测之外", model=REPO, revision=REV,
        task=task, instruction=record["instruction"], instruction_type="default", run=0, env_id=0,
        num_envs=1, env_seed=cfg["seed"], request_seed="uint32(crc32(env_seed:chunk_key))",
        success=good, terminal_condition="official_task_success" if good else "time_limit",
        simulator_reason=record.get("reason"), steps=steps, horizon_steps=horizon, dt=dt,
        simulation_seconds=round(steps*dt,4), policy_responses=len(chunks), hdf5_action_rows=len(commands),
        video_type="RoboLab 原始连续传感器录像", video=str(video.relative_to(run)), decoded_frames=frames,
        fps=fps, recorded_terminal_frame=(frames==steps), eval_output="rollout/episode_results.jsonl")
    columns = ["control_step_1based","chunk_start_0based","row_in_chunk","policy_request_seed",
               *[f"arm_joint_{i}" for i in range(1,8)],"gripper_command"]
    if args.check_only:
        assert summary == load(run/"summary.json")
        with (run/"actions.csv").open() as handle:
            reader = csv.reader(handle)
            assert next(reader) == columns and list(reader) == [[str(v) for v in row] for row in action_rows]
    else:
        with (run/"actions.csv").open("w", newline="") as handle:
            writer = csv.writer(handle);writer.writerow(columns);writer.writerows(action_rows)
        (run/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print(f"{task}: {steps}/{horizon} steps, {len(chunks)} real policy responses, terminal={summary['terminal_condition']}, "
          f"{frames} original sensor frames decoded at {fps:g} fps; all executed actions matched")


if __name__ == "__main__":
    main()
