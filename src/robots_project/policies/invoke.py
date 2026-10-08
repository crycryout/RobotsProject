"""Bounded policy invocation and fail-closed model conversion contract."""

from __future__ import annotations

import queue
import threading
import time

from robots_project.types import PolicyOutput


class PolicyError(RuntimeError):
    pass


class PolicyTimeoutError(PolicyError):
    """The worker may still be running; abandon the episode without concurrent close."""


def invoke(policy, observation, context, action_spec, timeout_s):
    specs = policy.specs()
    if specs.get("real_wam"):
        if not specs.get("action_conversion_confirmed"):
            raise PolicyError("Real-WAM conversion has not been confirmed by the model owner")
        if "diagnostic_state" in observation.policy or not observation.policy.get("rgb"):
            raise PolicyError("Visual WAM execution requires real sensor RGB without diagnostic state")
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
        raise PolicyTimeoutError(f"Policy timed out after {timeout_s} seconds") from error
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
    if policy.specs().get("real_wam"):
        conversion = output.conversion_metadata or {}
        if output.raw_model_output is None or conversion.get("target_action_spec_id") != action_spec.id or not conversion.get("confirmation_ref"):
            raise PolicyError("Real-WAM output must preserve raw model actions and conversion confirmation")
    if output.raw_model_output is not None:
        import numpy as np
        raw = output.raw_model_output
        if not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[0] != len(context["env_ids"]) or not np.isfinite(raw).all():
            raise PolicyError("Raw model actions require finite B x K_model x D_model shape")
    try:
        action_spec.validate(output.actions, len(context["env_ids"]), context["chunk_length"])
    except ValueError as error:
        raise PolicyError(str(error)) from error
    return output, time.perf_counter() - start
