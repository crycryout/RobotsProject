"""Bounded policy invocation and fail-closed model conversion contract."""

from __future__ import annotations

import queue
import threading
import time

from robots_project.types import PolicyOutput


class PolicyError(RuntimeError):
    pass


def invoke(policy, observation, context, action_spec, timeout_s):
    result = queue.Queue(maxsize=1)

    def worker():
        try:
            result.put((True, policy.act(observation, context)))
        except Exception as error:
            result.put((False, error))

    start = time.perf_counter()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        ok, output = result.get(timeout=timeout_s)
    except queue.Empty as error:
        raise PolicyError(f"Policy timed out after {timeout_s} seconds") from error
    if not ok:
        raise PolicyError(f"Policy invocation failed: {output}") from output
    thread.join()
    if not isinstance(output, PolicyOutput):
        raise PolicyError("Policy must return a PolicyOutput")
    if output.action_spec_id != action_spec.id:
        raise PolicyError("Action specification ID differs; semantic conversion is required")
    if output.request_id != context["request_id"]:
        raise PolicyError("Policy response request ID differs")
    if output.policy_version != policy.specs()["policy_version"]:
        raise PolicyError("Policy version changed inside a run")
    if policy.specs().get("real_wam") and not policy.specs().get("action_conversion_confirmed"):
        raise PolicyError("Real-WAM conversion has not been confirmed by the model owner")
    try:
        action_spec.validate(output.actions, len(context["env_ids"]), context["chunk_length"])
    except ValueError as error:
        raise PolicyError(str(error)) from error
    return output, time.perf_counter() - start
