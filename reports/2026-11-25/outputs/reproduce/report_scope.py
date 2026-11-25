"""Machine-readable scope of derived results. Original episode evidence is kept byte-for-byte."""
EVAL = {
    "evaluation_tasks": "10 pre-screened train-pool tasks excluded from v2 updates",
    "evaluation_instruction": "default",
    "official_test_split_evaluated": "no",
    "initial_state_evidence": "3 e03c1 arms: HDF5; Base: protocol keys only",
}
TRAIN = {"training_instructions": "default + specific", "training_window": "epochs 1-3"}


def annotate(name, row):
    meta = {"v2_training_seed": 20261123, "training_seeds_per_v2_arm": 1}
    if name in ("eval_tasks", "paired") or (name == "results" and row["dataset"] == "untrained_task_eval"):
        meta.update(EVAL)
    elif name in ("training_epochs", "training_tasks", "results"):
        meta.update(TRAIN)
    else:
        return row
    if name == "results":  # One schema for the two datasets. Empty values mark inapplicable fields.
        meta = {key:meta.get(key,"") for key in ("v2_training_seed","training_seeds_per_v2_arm",*TRAIN,*EVAL)}
    return {**row, **meta}
