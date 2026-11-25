"""Run exactly one real RoboLab episode, logging each seeded policy response before execution."""
import json
import os
import runpy
import time
import zlib
from pathlib import Path

import numpy as np


def main():
    # Let RoboLab itself select the task, construct Isaac Sim, and parse all CLI flags.
    robolab = Path(os.environ["ROBOLAB_ROOT"])
    entry = runpy.run_path(str(robolab / "policies/cosmos3/run.py"), run_name="robolab_entry")
    args = entry["args_cli"]
    app = entry["simulation_app"]
    try:
        if (args.num_envs, args.num_runs, len(args.task), args.instruction_type) != (1, 1, 1, ["default"]):
            raise ValueError("Expected one task, one env, one run, and the default instruction")
        if not Path(args.output_folder_name).is_absolute():
            raise ValueError("Use a fresh, absolute output directory")
        run_dir = Path(args.output_folder_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        traces = run_dir / "policy_chunks"
        traces.mkdir()
        config = {k: v for k, v in vars(args).items() if "token" not in k and
                  isinstance(v, (str, int, float, bool, list, type(None)))}
        (run_dir / "client_config.json").write_text(json.dumps(config, indent=2) + "\n")

        class LoggedClient(entry["Cosmos3Client"]):
            next_start = 0

            def _query_server(self, request):
                start = self.next_start
                key = f"{args.task[0]}_run0_env0_step{start:04d}"
                seed = zlib.crc32(f"{args.seed}:{key}".encode()) & 0xffffffff
                request = dict(request, seed=seed)
                begin = time.time()
                response = super()._query_server(request)
                raw = self._unpack_response(response)
                if raw.shape != (self.open_loop_horizon, 8) or not np.isfinite(raw).all():
                    raise ValueError(f"Invalid action chunk at {start}: {raw.shape}")
                np.savez_compressed(traces / f"{key}.npz", seed=np.int64(seed),
                    request_start_unix=begin, response_unix=time.time(), action=raw,
                    image=request["observation/image"], joint=request["observation/joint_position"],
                    gripper=request["observation/gripper_position"], prompt=request["prompt"])
                self.next_start += len(raw)
                return response

        entry["run_evaluation"](args, policy="cosmos3", client_factory=lambda a: LoggedClient(remote_uri=a.remote_uri))
    finally:
        app.close()


if __name__ == "__main__":
    main()
