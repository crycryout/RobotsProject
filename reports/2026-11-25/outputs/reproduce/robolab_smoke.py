"""Run the upstream H800 smoke tests in a new explicitly selected output directory."""
import argparse
from pathlib import Path
import pytest
from robolab.constants import get_output_dir, set_output_dir

parser=argparse.ArgumentParser()
parser.add_argument("out",type=Path)
parser.add_argument("--sim-only",action="store_true")
args=parser.parse_args()
out=args.out.resolve()
out.mkdir(parents=True,exist_ok=False)
set_output_dir(str(out))
assert Path(get_output_dir())==out
tests=Path("/home/robot_share/RoboLab/tests")
suite=[tests/"test_run_empty.py"]
if not args.sim_only:suite.extend([tests/"test_init_pose_layout.py",tests/"test_cosmos3_client.py"])
raise SystemExit(pytest.main(["-p","no:cacheprovider","-q",*map(str,suite),"--task","BananaInBowlTask",
                             f"--junitxml={out/'result.xml'}"]))
