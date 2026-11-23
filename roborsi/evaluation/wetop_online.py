# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run a local RoboRSI dialogue against the online WET-OP GenManip service.

The official client owns transport, retries, worker allocation and scoring.
This module translates public observations/actions and owns one dialogue per
native episode. It does not start a simulator or submit a leaderboard job.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from roborsi.agents.config import AgentConfig
from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action, Observation, StepResult, Task
from roborsi.evaluation.result import write_json_atomic
from roborsi.utils.config_values import json_value, reject_secrets

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "wet_op_bench_test"
# Pin the SDK because its wire protocol and episode lifecycle are part of the run.
CLIENT_REVISION = "fbfcd776639db82be3becdc6e39b5401f994da5c"
_PUBLIC_FIELDS = (
    "instruction",
    "state.joints",
    "state.gripper",
    "state.base",
    "state.ee_pose",
    "timestep",
    "reset",
    "episode_id",
    "robot_id",
)


def _url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (parsed.port is None or parsed.port > 0)
            and not any(c.isspace() or ord(c) < 32 for c in value)
            and "\\" not in value
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("URL must be HTTP(S), without credentials, query or fragment")
    return value.rstrip("/")


def _json(value: Any) -> Any:
    """Detach numpy values, rejecting non-finite and non-JSON payloads."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("native objects must use string keys")
        value = {key: _json(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_json(item) for item in value]
    json_value(value)
    return value


@dataclass(frozen=True)
class ModelProfile:
    """Resolved model settings; credentials are read only when building an Agent."""

    path: Path
    llm: dict[str, Any]
    api_key_env: str

    @classmethod
    def load(cls, path: Path) -> ModelProfile:
        path = path.expanduser().resolve()
        document = yaml.safe_load(path.read_text())
        allowed = {
            "schema_version",
            "provider",
            "base_url",
            "model_name",
            "base_url_env",
            "model_name_env",
            "api_key_env",
            "openai_format",
        }
        if not isinstance(document, dict) or set(document) - allowed:
            raise ValueError(
                "WET-OP requires an LLM model profile without VLA/tool services"
            )
        json_value(document)
        reject_secrets(document)
        if (
            type(document.get("schema_version")) is not int
            or document["schema_version"] != 1
        ):
            raise ValueError("model profile requires schema_version: 1")
        if document.get("provider") != "openai":
            raise ValueError("model profile must use an OpenAI-compatible provider")
        for name in ("base_url_env", "model_name_env", "api_key_env"):
            if name in document and (
                not isinstance(document[name], str)
                or not re.fullmatch(r"[A-Za-z_]\w*", document[name])
            ):
                raise ValueError(f"{name} must name an environment variable")
        model = os.environ.get(
            document.get("model_name_env", "MODEL_NAME"), document.get("model_name")
        )
        base_url = os.environ.get(
            document.get("base_url_env", "OPENAI_BASE_URL"), document.get("base_url")
        )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model_name must be nonempty (or set model_name_env)")
        if base_url is not None:
            if not isinstance(base_url, str):
                raise ValueError("base_url must be a URL or null")
            base_url = _url(base_url)
        api_format = document.get("openai_format", "chat")
        if api_format not in {None, "chat", "responses"}:
            raise ValueError("openai_format must be chat, responses or null")
        llm = {
            "provider": "openai",
            "model": model,
            "base_url": base_url,
            "parallel_tool_calls": False,
        }
        # The original OpenAI profile uses null to retain the Agent preset's
        # API choice. The WET-OP preset inherits Responses from the generic Agent.
        if api_format is not None:
            llm["openai_format"] = api_format
        return cls(
            path,
            llm,
            document.get("api_key_env", "OPENAI_API_KEY"),
        )

    def agent_config(self, path: Path, overrides: list[str]) -> AgentConfig:
        settings = [
            f"runtime.llm.{key}={json.dumps(value)}" for key, value in self.llm.items()
        ]
        config = AgentConfig.from_file(path, overrides=[*settings, *overrides])
        if config.factory != "roborsi.agents.config:make_embodied_agent":
            raise ValueError("online WET-OP requires the generic EmbodiedAgent factory")
        if config.document.get("agent", {}).get("planner", "api") != "api":
            raise ValueError("online WET-OP currently supports the API planner")
        runtime = config.runtime()  # Validate before allocating a remote worker.
        if runtime.llm is None:
            raise ValueError("online WET-OP requires runtime.llm")
        return config


def action_schema(max_chunk_steps: int = 60) -> dict[str, Any]:
    """Expose absolute Lift2 dual-arm targets, with bounded native repetitions."""
    if type(max_chunk_steps) is not int or not 1 <= max_chunk_steps <= 120:
        raise ValueError("max_chunk_steps must be between 1 and 120")

    def vector(length: int, description: str) -> dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "number"},
            "minItems": length,
            "maxItems": length,
            "description": description,
        }

    arm = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "position": vector(
                3, "Absolute XYZ in meters, in the same frame as state.ee_pose."
            ),
            "quaternion": vector(
                4,
                "Nonzero orientation quaternion in WXYZ order; normalized before submission.",
            ),
            "gripper": vector(
                2, "Absolute finger joint positions in native state.gripper units."
            ),
        },
        "required": ["position", "quaternion", "gripper"],
    }
    steps = {
        "type": "integer",
        "minimum": 1,
        "maximum": max_chunk_steps,
        "description": "Repeat this absolute target for this many native simulation steps, then observe. This is not interpolation.",
    }
    return {
        "type": "object",
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "control_type": {"const": "ee_pose"},
                    "left": arm,
                    "right": arm,
                    "steps": steps,
                },
                "required": ["control_type", "left", "right", "steps"],
            },
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "control_type": {"const": "joint_position"},
                    "positions": vector(
                        16,
                        "6 left arm joints, 2 left fingers, 6 right arm joints, 2 right fingers; absolute native joint units.",
                    ),
                    "steps": steps,
                },
                "required": ["control_type", "positions", "steps"],
            },
        ],
    }


def native_actions(action: Action, worker_id: str, max_chunk_steps: int) -> dict | list:
    """Validate a model tool call and convert it to official EvalClient input."""
    data = action.as_dict()["data"]
    Draft202012Validator(action_schema(max_chunk_steps)).validate(data)
    if data["control_type"] == "ee_pose":
        targets = []
        for side in ("left", "right"):
            arm = data[side]
            quat = np.asarray(arm["quaternion"], dtype=np.float64)
            norm = math.hypot(*quat)
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError("orientation quaternion must be finite and nonzero")
            targets.append([arm["position"], (quat / norm).tolist(), arm["gripper"]])
    else:
        targets = np.asarray(data["positions"], dtype=np.float64)
    native = {
        worker_id: {
            "action": targets,
            "base_motion": np.zeros(3, dtype=np.float64),
            "control_type": data["control_type"],
            "is_rel": False,
            "base_is_rel": True,
        }
    }
    return native if data["steps"] == 1 else [native for _ in range(int(data["steps"]))]


def observation(native: dict, channel: CaseMediaChannel, step: int) -> Observation:
    """Export public RGB/state only; evaluator metadata never enters the policy."""
    from PIL import Image

    episode_id = native.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("platform observation needs a native episode_id")
    if (
        not isinstance(native.get("instruction"), str)
        or not native["instruction"].strip()
    ):
        raise ValueError("platform observation needs a nonempty instruction")
    for field, shape in (("state.joints", (12,)), ("state.gripper", (4,))):
        array = np.asarray(native.get(field), dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"WET-OP Lift2 requires {field} with shape {shape}")
    poses = native.get("state.ee_pose")
    if not isinstance(poses, (list, tuple, np.ndarray)) or len(poses) != 2:
        raise ValueError("WET-OP Lift2 requires two state.ee_pose entries")
    for pose in poses:
        if (
            len(pose) != 2
            or np.asarray(pose[0]).shape != (3,)
            or np.asarray(pose[1]).shape != (4,)
        ):
            raise ValueError("state.ee_pose entries must be [XYZ, WXYZ]")
    data = {key: _json(native[key]) for key in _PUBLIC_FIELDS if key in native}
    media = []
    cameras = sorted(key for key in native if key.startswith("video."))
    if not cameras:
        raise ValueError("platform observation has no video camera frames")
    for key in cameras:
        frame = native[key]
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{key} must be an HWC uint8 RGB image")
            frame = Image.fromarray(frame)
        elif isinstance(frame, bytes):
            frame = Image.open(io.BytesIO(frame))
        if not isinstance(frame, Image.Image):
            raise ValueError(f"{key} is not a decoded camera image")
        buffer = io.BytesIO()
        frame.convert("RGB").save(buffer, format="PNG")
        media.append(channel.write(episode_id, step, buffer.getvalue(), "image/png"))
    data["media_fields"] = {key: index for index, key in enumerate(cameras)}
    result = Observation(
        benchmark=BENCHMARK,
        episode_id=episode_id,
        step_index=step,
        data=data,
        media=tuple(media),
    )
    channel.verify_observation(result)
    return result


class RunLog:
    """Record public observations, executed action requests and native results."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError(
                "output directory must be empty; use a new directory when resuming"
            )
        self.results: dict[str, dict] = {}
        self.metrics: dict = {}
        self.status = "running"
        self.decisions = 0
        self.event("run_started")

    def event(self, kind: str, **data: Any) -> None:
        event = {"time": datetime.now(timezone.utc).isoformat(), "type": kind, **data}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(_json(event), ensure_ascii=False, allow_nan=False) + "\n"
            )

    def result(self, value: dict) -> None:
        value = _json(value)
        identity = value.get("episode_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("episode_result needs an episode_id")
        for field in ("score", "sr"):
            number = value.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (float, int))
                or not 0 <= number <= 1
            ):
                raise ValueError(f"episode_result needs a native {field} in [0, 1]")
        if identity in self.results:
            if self.results[identity] != value:
                raise ValueError("conflicting results for the same native episode")
            return
        self.results[identity] = value
        self.event("episode_result", result=value)

    def save(self) -> dict:
        result = {
            "status": self.status,
            "completed": self.status == "completed",
            "decisions": self.decisions,
            "scored_episodes": len(self.results),
            "episode_results": list(self.results.values()),
            "platform_metrics": self.metrics,
        }
        write_json_atomic(self.directory / "summary.json", result)
        return result


@dataclass
class Episode:
    identity: str
    agent: Any
    channel: CaseMediaChannel
    directory: Path
    decisions: int = 0


_CONTROL_PROMPT = """
You control the Lift2 dual-arm robot in the online WET-OP scientific experiments.
Use only the provided instruction, RGB camera images and measured robot state.
Each roborsi_action sets absolute targets for BOTH arms. XYZ is in meters in
the same frame as state.ee_pose; quaternions are WXYZ. Finger targets use native
state.gripper joint units, not a normalized open/close command. Use the measured
pose and finger values to hold an arm stationary. state.joints has 12 arm joints;
state.gripper has 4 finger joints. Joint-position targets interleave 6 left arm
joints, 2 left fingers, 6 right arm joints, 2 right fingers. The mobile base is held
stationary. steps repeats the target for bounded native simulation steps, then
returns camera feedback; it is not a duration in seconds or interpolated motion.
Use short motions, inspect the feedback, and adjust. The server may end a chunk
early. The platform alone determines success, failure and episode completion.
"""


def agent_factory(config: AgentConfig, profile: ModelProfile) -> Callable:
    """Reuse the project's configured EmbodiedAgent and persistent tool loop."""

    def build(task: Task, directory: Path, channel: CaseMediaChannel):
        from roborsi.embodied_agent import EmbodiedAgent
        from roborsi.evaluation.embodied_bridge import (
            _EmbodiedToolConnector,
            roborsi_action_tool,
        )

        planner = config.create_agent(directory)
        if (
            not isinstance(planner, EmbodiedAgent)
            or planner.runtime is None
            or planner.runtime.llm is None
        ):
            raise ValueError(
                "online evaluation requires an API EmbodiedAgent with runtime.llm"
            )
        key = os.environ.get(profile.api_key_env)
        if not key:
            raise ValueError(f"set {profile.api_key_env} before running")
        planner = replace(
            planner,
            runtime=replace(
                planner.runtime, llm=replace(planner.runtime.llm, api_key=key)
            ),
            local_tools=(
                *planner.local_tools,
                roborsi_action_tool(task.capabilities["action_schema"]),
            ),
        )
        connector = _EmbodiedToolConnector(
            planner,
            media_dir=channel.media_dir,
            system_prompt=planner.system_prompt + "\n" + _CONTROL_PROMPT,
            output_dir=directory,
            action_timeout_s=planner.action_timeout_s,
            invalid_action_retries=planner.invalid_action_retries,
            skills=planner.evaluation_skills,
            skill_paths=planner.evaluation_skill_paths,
            memory=planner.evaluation_memory,
            initial_context=planner.evaluation_initial_context,
        )
        connector.reset(task)
        return connector

    return build


def run_online(
    client: Any,
    build_agent: Callable,
    directory: Path,
    *,
    worker_id: str = "0",
    max_chunk_steps: int = 60,
    max_decisions: int = 1000,
    max_episodes: int | None = None,
    run_metadata: dict | None = None,
) -> dict:
    """Drive one owned worker; SDK done means the whole job, not one episode."""
    try:
        schema = action_schema(max_chunk_steps)
        if type(max_decisions) is not int or max_decisions < 1:
            raise ValueError("max_decisions must be positive")
        if max_episodes is not None and (
            type(max_episodes) is not int or max_episodes < 1
        ):
            raise ValueError("max_episodes must be positive")
        log = RunLog(directory)
    except BaseException:
        client.close()
        raise
    active: Episode | None = None
    attempts = 0

    def close_active() -> None:
        nonlocal active
        previous, active = active, None
        if previous is not None:
            previous.agent.close()

    def end_active(result: dict | None, reason: str) -> None:
        if active is None:
            return
        try:
            if result is not None and active.decisions:
                # Auto-reset responses contain the NEXT scene. Never label that
                # scene's images as terminal feedback from the previous one.
                terminal = Observation(
                    benchmark=BENCHMARK,
                    episode_id=active.identity,
                    step_index=active.decisions,
                    data={"terminal_observation_available": False},
                )
                active.agent.observe_end(
                    StepResult(
                        observation=terminal,
                        terminated=True,
                        truncated=False,
                        termination_reason="platform_episode_ended",
                        reward=float(result["score"]),
                        native={"episode_result": result},
                    )
                )
            log.event("dialogue_ended", episode_id=active.identity, reason=reason)
        finally:
            close_active()

    try:
        with ExitStack() as cleanup:
            cleanup.callback(client.close)
            cleanup.callback(close_active)
            if run_metadata is not None:
                write_json_atomic(log.directory / "run.json", run_metadata)
            response = client.reset()
            done = False
            while True:
                if not isinstance(response, dict) or not isinstance(
                    response.get(worker_id), dict
                ):
                    raise ValueError("platform response is missing the owned worker")
                worker = response[worker_id]
                if worker.get("reset_pending"):
                    raise ValueError(
                        "SDK returned unresolved reset_pending; check the pinned client version"
                    )
                result = worker.get("episode_result")
                native = worker.get("obs")
                if native is not None and not isinstance(native, dict):
                    raise ValueError("worker observation must be an object")
                if worker.get("metric") is not None:
                    log.metrics = _json(worker["metric"])
                if result is not None:
                    if not isinstance(result, dict):
                        raise ValueError("episode_result must be an object")
                    if (
                        active is not None
                        and result.get("episode_id") != active.identity
                        and result.get("episode_id") not in log.results
                    ):
                        raise ValueError("result does not match the active episode")
                    log.result(result)
                    if active is not None and result["episode_id"] == active.identity:
                        end_active(_json(result), "native_result")

                if active is not None and (
                    native is None
                    or worker.get("lock_lost")
                    or native.get("episode_id") != active.identity
                    or native.get("reset")
                ):
                    end_active(
                        None,
                        "lock_lost" if worker.get("lock_lost") else "unscored_episode",
                    )
                    if not worker.get("lock_lost"):
                        raise RuntimeError(
                            "platform changed episode without returning its native result"
                        )
                if native is None:
                    if worker.get("metric") is None:
                        raise ValueError(
                            "empty observation without final platform metrics"
                        )
                    log.status = "completed"
                    break
                if done:
                    raise ValueError(
                        "SDK reported a finished job with a live observation"
                    )
                if max_episodes is not None and len(log.results) >= max_episodes:
                    log.status = "limited"
                    break
                if active is None:
                    attempts += 1
                    identity = native.get("episode_id")
                    if not isinstance(identity, str) or not identity:
                        raise ValueError("observation needs a native episode_id")
                    if identity in log.results:
                        raise ValueError("platform returned a completed episode again")
                    tag = hashlib.sha256(identity.encode()).hexdigest()[:12]
                    episode_dir = log.directory / "episodes" / f"{attempts:06d}-{tag}"
                    episode_dir.mkdir(parents=True)
                    channel = CaseMediaChannel(episode_dir / "media")
                    # Validate the first observation before constructing a planner.
                    current = observation(native, channel, 0)
                    suffix = identity.rsplit("/", 1)[-1]
                    task = Task(
                        benchmark=BENCHMARK,
                        task_id=identity,
                        seed=int(suffix)
                        if suffix.isascii() and suffix.isdecimal()
                        else 0,
                        instruction=native["instruction"],
                        max_steps=max_decisions,
                        native={
                            "episode_id": identity,
                            "seed_source": "episode_id_suffix"
                            if suffix.isascii() and suffix.isdecimal()
                            else "not_exposed_placeholder",
                            "worker_id": worker_id,
                            "max_steps_unit": "agent_decisions",
                        },
                        capabilities={"action_schema": schema},
                    )
                    write_json_atomic(episode_dir / "task.json", task.as_dict())
                    active = Episode(
                        identity,
                        build_agent(task, episode_dir, channel),
                        channel,
                        episode_dir,
                    )
                    log.event(
                        "dialogue_started",
                        episode_id=identity,
                        directory=episode_dir.relative_to(log.directory).as_posix(),
                    )
                else:
                    current = observation(native, active.channel, active.decisions)
                if active.decisions >= max_decisions:
                    raise RuntimeError(
                        "local decision limit reached before a native episode result"
                    )
                log.event(
                    "observation",
                    directory=active.directory.relative_to(log.directory).as_posix(),
                    observation=current.as_dict(),
                )
                action = active.agent.act(current)
                if not isinstance(action, Action) or (
                    action.benchmark,
                    action.episode_id,
                    action.step_index,
                ) != (BENCHMARK, active.identity, active.decisions):
                    raise ValueError(
                        "Agent action does not match the current observation"
                    )
                request = native_actions(action, worker_id, max_chunk_steps)
                log.event("action_requested", action=action.as_dict())
                # Do not add an outer retry: only the official SDK can replay
                # physical actions safely using its reliable-request protocol.
                response, done = client.step(request)
                active.decisions += 1
                log.decisions += 1
                log.save()
        return log.save()
    except BaseException as exc:
        log.status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        log.event("run_error", error_type=type(exc).__name__)
        log.save()
        raise


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv: list[str] | None = None) -> int:
    """Validate configuration offline or attach to an already allocated job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument(
        "--model-config",
        type=Path,
        default=ROOT / "configs/models/volcengine_agent_plan.yaml",
    )
    parser.add_argument(
        "--agent-config", type=Path, default=ROOT / "configs/agents/wetop/online.yaml"
    )
    parser.add_argument("--agent-set", action="append", default=[], metavar="PATH=JSON")
    parser.add_argument("--endpoint", default=os.environ.get("EBENCH_ONLINE_ENDPOINT"))
    parser.add_argument("--run-id", default=os.environ.get("EBENCH_TASK_ID"))
    parser.add_argument("--token-env", default="EBENCH_SUBMIT_TOKEN")
    parser.add_argument("--worker-id", choices=[str(i) for i in range(8)], default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-chunk-steps", type=_positive, default=60)
    parser.add_argument(
        "--max-decisions",
        type=_positive,
        default=1000,
        help="per episode; hitting the limit stops the client without inventing a score",
    )
    parser.add_argument(
        "--max-episodes",
        type=_positive,
        help="stop after this many native results; the remote job may remain incomplete",
    )
    args = parser.parse_args(argv)
    try:
        profile = ModelProfile.load(args.model_config)
        config = profile.agent_config(args.agent_config, args.agent_set)
        action_schema(args.max_chunk_steps)
        if not re.fullmatch(r"[A-Za-z_]\w*", args.token_env):
            raise ValueError("token-env must name an environment variable")
        try:
            sdk_version = importlib.metadata.version("genmanip-client")
        except importlib.metadata.PackageNotFoundError:
            sdk_version = None
        if args.command == "check":
            print(
                json.dumps(
                    {
                        "benchmark": BENCHMARK,
                        "model": config.document["runtime"]["llm"],
                        "agent_config_sha256": config.fingerprint,
                        "model_key_env": profile.api_key_env,
                        "model_key_present": bool(os.environ.get(profile.api_key_env)),
                        "platform_token_present": bool(os.environ.get(args.token_env)),
                        "client_version": sdk_version,
                        "expected_client_revision": CLIENT_REVISION,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if not args.endpoint or not args.run_id:
            raise ValueError(
                "set --endpoint and --run-id from gmp online submit (or EBENCH_ONLINE_ENDPOINT/EBENCH_TASK_ID)"
            )
        endpoint = _url(args.endpoint)
        if not os.environ.get(args.token_env) or not os.environ.get(
            profile.api_key_env
        ):
            raise ValueError(
                f"set both {args.token_env} and {profile.api_key_env} before running"
            )
        try:
            from genmanip_client import EvalClient
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "GenManip client is unavailable; install .[runtime,wetop] and libturbojpeg (see docs/wetop-online.zh-CN.md)"
            ) from exc
        tag = hashlib.sha256(args.run_id.encode()).hexdigest()[:12]
        directory = (
            args.output_dir
            or ROOT
            / "work/wetop-online"
            / tag
            / f"worker-{args.worker_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
        )
        # Reserve the output before allocating a worker, so a typo cannot reset
        # the remote worker and then fail because an old output directory exists.
        directory = directory.expanduser().resolve()
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("output directory must be empty")
        directory.mkdir(parents=True, exist_ok=True)
        client = EvalClient(
            base_url=endpoint,
            token=os.environ[args.token_env],
            run_id=args.run_id,
            worker_ids=[args.worker_id],
            save_result=False,
            save_process=False,
        )
        metadata = {
            "benchmark": BENCHMARK,
            "run_id": args.run_id,
            "endpoint": endpoint,
            "worker_id": args.worker_id,
            "client_version": sdk_version,
            "expected_client_revision": CLIENT_REVISION,
            "agent_config_sha256": config.fingerprint,
            "model_profile_sha256": hashlib.sha256(
                profile.path.read_bytes()
            ).hexdigest(),
            "max_chunk_steps": args.max_chunk_steps,
            "max_decisions": args.max_decisions,
        }
        result = run_online(
            client,
            agent_factory(config, profile),
            directory,
            worker_id=args.worker_id,
            max_chunk_steps=args.max_chunk_steps,
            max_decisions=args.max_decisions,
            max_episodes=args.max_episodes,
            run_metadata=metadata,
        )
        print(
            json.dumps(
                {"output_dir": str(directory), **result}, ensure_ascii=False, indent=2
            )
        )
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        message = str(exc)
        for name in (
            args.token_env,
            profile.api_key_env if "profile" in locals() else "ARK_API_KEY",
        ):
            secret = os.environ.get(name)
            if secret:
                message = message.replace(secret, "[redacted]")
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
