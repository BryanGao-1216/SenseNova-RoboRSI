# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Online adapter contracts without platform credentials or a simulator."""

from __future__ import annotations

import base64
import io
import json
import pickle
import sys
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from urllib.parse import urlsplit

import numpy as np
import pytest
from jsonschema import ValidationError
from PIL import Image

from roborsi.evaluation.media import CaseMediaChannel
from roborsi.evaluation.messages import Action
from roborsi.evaluation.wetop_online import (
    BENCHMARK,
    ROOT,
    ModelProfile,
    action_schema,
    agent_factory,
    main,
    native_actions,
    observation,
    run_online,
)

EP1 = "wet_op_bench_test/run/scientific_experiments/pour/test_mixed_1/000"
EP2 = "wet_op_bench_test/run/scientific_experiments/pour/test_mixed_1/001"


def native(identity=EP1, *, reset=True, timestep=0):
    return {
        "episode_id": identity,
        "instruction": "Pour into the cylinder.",
        "state.joints": np.arange(12, dtype=np.float32),
        "state.gripper": np.array([0.02, 0.02, 0.03, 0.03]),
        "state.base": np.zeros(3),
        "state.ee_pose": [
            ([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0]),
            ([0.4, 0.5, 0.6], [1.0, 0.0, 0.0, 0.0]),
        ],
        "video.overlook_view": np.full((4, 6, 3), 17, dtype=np.uint8),
        "video.left_view": np.full((4, 6, 3), 42, dtype=np.uint8),
        "timestep": timestep,
        "reset": reset,
        "robot_id": "manip/lift2/R5a",
        "private_goal_pose": [9.0, 9.0, 9.0],
        "evaluator": {"score": 1},
    }


def result(identity=EP1, score=0.4, sr=0.0):
    return {
        "episode_id": identity,
        "score": score,
        "sr": sr,
        "task_name": "scientific_experiments/pour/test_mixed_1",
        "seed": identity[-3:],
    }


def response(obs, episode_result=None, **fields):
    return {
        "0": {
            "obs": obs,
            "metric": {"pour": {"score": 0.4}},
            "episode_result": episode_result,
            **fields,
        }
    }


def joint_action(steps=2):
    return {"control_type": "joint_position", "positions": [0.0] * 16, "steps": steps}


def ee_action(steps=2):
    arm = {
        "position": [0.1, 0.2, 0.3],
        "quaternion": [2.0, 0.0, 0.0, 0.0],
        "gripper": [0.02, 0.02],
    }
    return {
        "control_type": "ee_pose",
        "left": arm,
        "right": deepcopy(arm),
        "steps": steps,
    }


def action(data, episode_id=EP1, step=0):
    return Action(
        benchmark=BENCHMARK, episode_id=episode_id, step_index=step, data=data
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = 0

    def reset(self):
        return next(self.responses)

    def step(self, request):
        self.requests.append(request)
        item = next(self.responses)
        if isinstance(item, BaseException):
            raise item
        return item, item["0"]["obs"] is None

    def close(self):
        self.closed += 1


class FakeAgent:
    def __init__(self, task, directory, channel):
        self.task, self.directory, self.channel = task, directory, channel
        self.observations, self.ends = [], []
        self.closed = 0

    def act(self, obs):
        self.observations.append(obs)
        return action(joint_action(), obs.episode_id, obs.step_index)

    def observe_end(self, step):
        self.ends.append(step)

    def close(self):
        self.closed += 1


def fake_factory(agents):
    def build(*args):
        agent = FakeAgent(*args)
        agents.append(agent)
        return agent

    return build


def test_observation_keeps_rgb_names_and_public_state_only(tmp_path):
    channel = CaseMediaChannel(tmp_path)
    obs = observation(native(), channel, 0)
    assert obs.data["media_fields"] == {"video.left_view": 0, "video.overlook_view": 1}
    assert len(obs.media) == 2
    assert "private_goal_pose" not in obs.data and "evaluator" not in obs.data
    assert obs.data["state.joints"] == list(range(12))
    for index, expected in enumerate([42, 17]):
        channel.verify_observation(obs)
        with Image.open(
            channel.media_dir / Path(obs.media[index].ref).relative_to("media")
        ) as image:
            assert image.getpixel((0, 0)) == (expected,) * 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda obs: obs.update({"state.joints": np.zeros(7)}),
        lambda obs: obs.update({"state.gripper": np.zeros(2)}),
        lambda obs: obs.update({"state.ee_pose": []}),
        lambda obs: obs.update({"instruction": ""}),
        lambda obs: obs.update({"episode_id": ""}),
        lambda obs: obs.update({"state.base": [float("nan"), 0, 0]}),
        lambda obs: obs.update({"video.left_view": np.zeros((4, 6))}),
        lambda obs: [obs.pop(key) for key in list(obs) if key.startswith("video.")],
    ],
)
def test_invalid_observation_fails_before_model(mutation, tmp_path):
    obs = native()
    mutation(obs)
    with pytest.raises((ValueError, TypeError)):
        observation(obs, CaseMediaChannel(tmp_path), 0)


def test_absolute_dual_arm_ee_pose_chunk_uses_native_conventions():
    chunk = native_actions(action(ee_action(3)), "7", 60)
    assert len(chunk) == 3 and set(chunk[0]) == {"7"}
    payload = chunk[0]["7"]
    assert payload["control_type"] == "ee_pose"
    assert payload["is_rel"] is False and payload["base_is_rel"] is True
    np.testing.assert_array_equal(payload["base_motion"], [0.0, 0.0, 0.0])
    assert payload["action"][0] == [[0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0], [0.02, 0.02]]
    assert len(payload["action"]) == 2


def test_joint_positions_preserve_interleaved_finger_indices():
    data = joint_action(1)
    data["positions"] = list(range(16))
    converted = native_actions(action(data), "0", 60)
    assert isinstance(converted, dict)
    np.testing.assert_array_equal(converted["0"]["action"], np.arange(16))


@pytest.mark.parametrize(
    "data",
    [
        joint_action(0),
        joint_action(61),
        {**joint_action(), "positions": [0.0] * 7},
        {**joint_action(), "base_motion": [1, 1, 1]},
    ],
)
def test_bad_actions_never_reach_transport(data):
    with pytest.raises(ValidationError):
        native_actions(action(data), "0", 60)


def test_zero_quaternion_is_rejected():
    data = ee_action()
    data["left"]["quaternion"] = [0.0] * 4
    with pytest.raises(ValueError, match="quaternion"):
        native_actions(action(data), "0", 60)


def test_rollover_resets_dialogue_and_preserves_native_partial_scores(tmp_path):
    client = FakeClient(
        [
            response(native()),
            response(native(reset=False, timestep=2)),
            response(native(EP2), result()),
            response(None, result(EP2, 1.0, 1.0)),
        ]
    )
    agents = []
    summary = run_online(
        client, fake_factory(agents), tmp_path, run_metadata={"run_id": "run"}
    )
    assert summary["status"] == "completed" and summary["scored_episodes"] == 2
    assert [item["score"] for item in summary["episode_results"]] == [0.4, 1.0]
    assert [item["sr"] for item in summary["episode_results"]] == [0.0, 1.0]
    assert [obs.step_index for obs in agents[0].observations] == [0, 1]
    assert [obs.step_index for obs in agents[1].observations] == [0]
    assert len(agents) == 2 and client.closed == 1
    assert all(agent.closed == 1 for agent in agents)
    old_end = agents[0].ends[0]
    assert old_end.observation.episode_id == EP1
    assert old_end.observation.step_index == 2 and old_end.observation.media == ()
    assert old_end.observation.data == {"terminal_observation_available": False}
    assert old_end.native["episode_result"]["score"] == 0.4
    assert json.loads((tmp_path / "run.json").read_text()) == {"run_id": "run"}
    assert len(client.requests) == 3 and all(
        len(chunk) == 2 for chunk in client.requests
    )


def test_real_embodied_agent_loop_sees_fresh_context_after_rollover(
    tmp_path, monkeypatch
):
    from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
    from pydantic_ai.models.function import FunctionModel

    from roborsi.llm import LLMConfig

    turns = []

    def model(messages, info):
        assert {tool.name for tool in info.function_tools} == {"roborsi_action"}
        returned = sum(
            isinstance(part, ToolReturnPart)
            for message in messages
            for part in message.parts
        )
        turns.append(returned)
        assert "private_goal_pose" not in str(messages)
        return ModelResponse(parts=[ToolCallPart("roborsi_action", ee_action())])

    monkeypatch.setattr(LLMConfig, "build_model", lambda self: FunctionModel(model))
    monkeypatch.setenv("ARK_API_KEY", "offline-fake-key")
    profile = ModelProfile.load(ROOT / "configs/models/volcengine_agent_plan.yaml")
    config = profile.agent_config(ROOT / "configs/agents/wetop/online.yaml", [])
    client = FakeClient(
        [
            response(native()),
            response(native(reset=False, timestep=2)),
            response(native(EP2), result()),
            response(None, result(EP2, 1.0, 1.0)),
        ]
    )
    summary = run_online(client, agent_factory(config, profile), tmp_path)
    assert summary["completed"] is True and turns == [0, 1, 0]
    for config_path in tmp_path.glob("episodes/*/agent-config.json"):
        assert "offline-fake-key" not in config_path.read_text()
    assert client.closed == 1


def test_limit_stops_without_starting_next_model_or_claiming_completion(tmp_path):
    agents = []
    client = FakeClient([response(native()), response(native(EP2), result())])
    summary = run_online(client, fake_factory(agents), tmp_path, max_episodes=1)
    assert summary["status"] == "limited" and summary["completed"] is False
    assert summary["scored_episodes"] == 1 and len(agents) == 1
    assert client.closed == 1 and agents[0].closed == 1


def test_last_episode_at_limit_still_reports_actual_job_completion(tmp_path):
    client = FakeClient([response(native()), response(None, result())])
    summary = run_online(client, fake_factory([]), tmp_path, max_episodes=1)
    assert summary["completed"] is True


@pytest.mark.parametrize(
    "failure", [RuntimeError("transport failed"), KeyboardInterrupt()]
)
def test_failed_step_is_not_retried_and_resources_are_closed(tmp_path, failure):
    agents = []
    client = FakeClient([response(native()), failure])
    with pytest.raises(type(failure)):
        run_online(client, fake_factory(agents), tmp_path)
    assert len(client.requests) == 1 and client.closed == agents[0].closed == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] is False and summary["scored_episodes"] == 0


def test_local_decision_budget_does_not_invent_native_failure(tmp_path):
    client = FakeClient([response(native()), response(native(reset=False))])
    agents = []
    with pytest.raises(RuntimeError, match="decision limit"):
        run_online(client, fake_factory(agents), tmp_path, max_decisions=1)
    assert len(client.requests) == 1
    assert json.loads((tmp_path / "summary.json").read_text())["episode_results"] == []
    assert not agents[0].ends


def test_unscored_rollover_is_an_error_without_fabricated_score(tmp_path):
    client = FakeClient([response(native()), response(native(EP2))])
    agents = []
    with pytest.raises(RuntimeError, match="without returning"):
        run_online(client, fake_factory(agents), tmp_path)
    assert len(agents) == 1 and agents[0].closed == 1
    assert json.loads((tmp_path / "summary.json").read_text())["episode_results"] == []


def test_lock_loss_restarts_context_without_scoring_old_attempt(tmp_path):
    client = FakeClient(
        [
            response(native()),
            response(native(EP2), lock_lost=True),
            response(None, result(EP2, 1.0, 1.0)),
        ]
    )
    agents = []
    summary = run_online(client, fake_factory(agents), tmp_path)
    assert len(agents) == 2 and not agents[0].ends
    assert summary["scored_episodes"] == 1
    assert all(agent.closed == 1 for agent in agents)


def test_mismatched_result_does_not_finish_another_episode(tmp_path):
    client = FakeClient([response(native()), response(native(EP2), result(EP2))])
    with pytest.raises(ValueError, match="does not match"):
        run_online(client, fake_factory([]), tmp_path)
    assert json.loads((tmp_path / "summary.json").read_text())["episode_results"] == []


def test_no_tasks_left_on_reset_needs_no_model(tmp_path):
    agents = []
    summary = run_online(FakeClient([response(None)]), fake_factory(agents), tmp_path)
    assert summary["completed"] is True and not agents


def test_existing_output_is_not_overwritten(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text("keep")
    client = FakeClient([])
    with pytest.raises(ValueError, match="empty"):
        run_online(client, fake_factory([]), tmp_path)
    assert path.read_text() == "keep" and client.closed == 1


def test_profile_environment_overrides_and_no_key_persistence(monkeypatch):
    monkeypatch.setenv("ARK_MODEL_NAME", "my-kimi-endpoint")
    monkeypatch.setenv("ARK_BASE_URL", "https://example.org/api/plan/v3/")
    monkeypatch.setenv("ARK_API_KEY", "never-save-this")
    profile = ModelProfile.load(ROOT / "configs/models/volcengine_agent_plan.yaml")
    config = profile.agent_config(ROOT / "configs/agents/wetop/online.yaml", [])
    assert profile.llm["model"] == "my-kimi-endpoint"
    assert profile.llm["base_url"] == "https://example.org/api/plan/v3"
    assert "never-save-this" not in config.document_json
    assert config.document["runtime"]["context"]["keep_turns"] == 8


def test_offline_check_reports_missing_credentials_without_network(monkeypatch, capsys):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    monkeypatch.delenv("EBENCH_SUBMIT_TOKEN", raising=False)
    assert main(["check"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert not data["model_key_present"] and not data["platform_token_present"]


def test_original_openai_profile_preserves_agent_responses_format(monkeypatch):
    monkeypatch.delenv("GPT6_MODEL_NAME", raising=False)
    monkeypatch.delenv("GPT6_BASE_URL", raising=False)
    profile = ModelProfile.load(ROOT / "configs/models/gpt6.yaml")
    config = profile.agent_config(ROOT / "configs/agents/wetop/online.yaml", [])
    llm = config.runtime().llm
    assert profile.api_key_env == "OPENAI_API_KEY"
    assert llm.model == "gpt-6-astra"
    assert llm.base_url == "https://api.openai.com/v1"
    assert llm.openai_format == "responses"


def test_live_run_requires_endpoint_and_credentials(capsys):
    assert main(["run"]) == 1
    assert "endpoint" in capsys.readouterr().err


@pytest.mark.parametrize("error_type", [ImportError, RuntimeError, AttributeError])
def test_client_import_error_shows_cause_and_interpreter_without_secrets(
    monkeypatch, capsys, error_type
):
    module = ModuleType("genmanip_client")

    def missing(name):
        if name == "EvalClient":
            raise error_type("native JPEG import failed offline-platform-token")
        raise AttributeError(name)

    module.__getattr__ = missing
    monkeypatch.setitem(sys.modules, "genmanip_client", module)
    monkeypatch.setenv("ARK_API_KEY", "offline-model-key")
    monkeypatch.setenv("EBENCH_SUBMIT_TOKEN", "offline-platform-token")
    assert (
        main(
            [
                "run",
                "--endpoint",
                "https://example.org/endpoint",
                "--run-id",
                "offline-run",
            ]
        )
        == 1
    )
    message = capsys.readouterr().err
    assert sys.executable in message
    assert f"{error_type.__name__}: native JPEG import failed" in message
    assert "PyTurboJPEG<2" in message
    assert "offline-platform-token" not in message
    assert "[redacted]" in message


def test_chunk_limit_is_validated_before_any_remote_reset():
    with pytest.raises(ValueError, match="between"):
        action_schema(121)


def test_kimi_chat_request_carries_images_schema_and_tool_feedback(
    tmp_path, monkeypatch
):
    import httpx
    import pydantic_ai.providers.openai as providers

    original_provider = providers.OpenAIProvider
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/api/plan/v3/chat/completions"
        assert request.headers["authorization"] == "Bearer offline-fake-key"
        assert body["model"] == "kimi-k3.1-pro"
        assert body["parallel_tool_calls"] is False
        tool = body["tools"][0]["function"]
        assert tool["name"] == "roborsi_action"
        assert tool["parameters"]["type"] == "object"
        assert "private_goal_pose" not in json.dumps(body)
        call = {
            "id": f"call_{len(requests)}",
            "type": "function",
            "function": {
                "name": "roborsi_action",
                "arguments": json.dumps(ee_action()),
            },
        }
        if body.get("stream"):
            delta = {"role": "assistant", "tool_calls": [{"index": 0, **call}]}
            chunks = [
                {
                    "id": "chatcmpl-offline",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                },
                {
                    "id": "chatcmpl-offline",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body["model"],
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                },
            ]
            wire = (
                "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, text=wire, headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [call],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    def provider(**kwargs):
        return original_provider(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
        )

    monkeypatch.setattr(providers, "OpenAIProvider", provider)
    monkeypatch.setenv("ARK_API_KEY", "offline-fake-key")
    monkeypatch.delenv("ARK_MODEL_NAME", raising=False)
    monkeypatch.delenv("ARK_BASE_URL", raising=False)
    profile = ModelProfile.load(ROOT / "configs/models/volcengine_agent_plan.yaml")
    config = profile.agent_config(ROOT / "configs/agents/wetop/online.yaml", [])
    client = FakeClient(
        [
            response(native()),
            response(native(reset=False, timestep=2)),
            response(None, result()),
        ]
    )
    assert run_online(client, agent_factory(config, profile), tmp_path)["completed"]
    assert len(requests) == 2
    image_count = sum(
        part.get("type") == "image_url"
        for message in requests[0]["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )
    assert image_count == 2
    assert any(message["role"] == "tool" for message in requests[1]["messages"])
    for path in tmp_path.rglob("*"):
        if path.suffix in {".json", ".jsonl"}:
            assert "offline-fake-key" not in path.read_text()


def test_official_sdk_reliable_http_reset_poll_and_early_chunk_end(
    tmp_path, monkeypatch
):
    pytest.importorskip("genmanip_client")
    pytest.importorskip("cv2")
    turbojpeg = pytest.importorskip("turbojpeg")

    class PillowDecoder:
        # Exercise the SDK wire format on hosts without libturbojpeg. Only the
        # native JPEG decoder is substituted; EvalClient/transport are upstream.
        def decode(self, data, **kwargs):
            with Image.open(io.BytesIO(data)) as image:
                return np.array(image.convert("RGB"))

    monkeypatch.setattr(turbojpeg, "TurboJPEG", PillowDecoder)
    from genmanip_client import EvalClient

    monkeypatch.setenv("GENMANIP_RESULT_DIR", str(tmp_path / "sdk-output"))
    executed = []
    replay = {}
    posted = []
    kills = []

    def encode(value):
        if isinstance(value, np.ndarray):
            if value.dtype == np.uint8 and value.ndim == 3:
                buffer = io.BytesIO()
                Image.fromarray(value).save(buffer, format="JPEG")
                return {"type": "jpeg_bytes", "data": buffer.getvalue()}
            return {
                "type": "numpy_array",
                "dtype": str(value.dtype),
                "shape": value.shape,
                "data": base64.b64encode(value.tobytes()).decode(),
            }
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        return value

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, status=200, binary=False):
            body = pickle.dumps(data) if binary else json.dumps(data).encode()
            self.send_response(status)
            self.send_header(
                "Content-Type",
                "application/octet-stream" if binary else "application/json",
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlsplit(self.path).path == "/request_protocol":
                self.respond(
                    {
                        "protocol": "genmanip-replay-v1",
                        "epoch": "test-epoch",
                        "worker_generations": {"0": 0},
                    }
                )
            else:
                self.respond({})

        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer offline-platform-token"
            wire = self.rfile.read(int(self.headers["Content-Length"]))
            if urlsplit(self.path).path == "/kill":
                kills.append(json.loads(wire))
                self.respond({})
                return
            envelope = pickle.loads(wire)
            posted.append(envelope)
            sequence = envelope["sequence"]
            if sequence not in replay:
                operation = envelope["operation"]
                executed.append(operation)
                if operation == "reset":
                    output = response({"reset": True}, reset_pending=True)
                elif operation == "reset_result":
                    output = encode(response(native()))
                else:
                    assert operation == "step_chunk"
                    chunk = pickle.loads(envelope["payload"])
                    assert len(chunk) == 2 and set(chunk[0]) == {"0"}
                    np.testing.assert_array_equal(chunk[0]["0"]["action"], np.zeros(16))
                    output = {
                        "obs": encode(response(native(EP2), result()))
                        if executed.count("step_chunk") == 1
                        else response(None, result(EP2, 1.0, 1.0)),
                        "executed_steps": 1,
                    }
                replay[sequence] = {
                    "protocol": "genmanip-replay-v1",
                    "state": "completed",
                    "status_code": 200,
                    "body": pickle.dumps(output),
                    **{
                        key: envelope[key] for key in ("epoch", "stream_id", "sequence")
                    },
                }
                if operation == "step_chunk" and executed.count("step_chunk") == 1:
                    self.respond(
                        {**replay[sequence], "state": "pending"},
                        status=202,
                        binary=True,
                    )
                    return
            self.respond(replay[sequence], binary=True)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = EvalClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            token="offline-platform-token",
            worker_ids=["0"],
            run_id="offline-run",
            verbose=False,
            save_result=False,
            save_process=False,
        )
        agents = []
        summary = run_online(client, fake_factory(agents), tmp_path / "roborsi")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert summary["completed"] and len(agents) == 2
    assert executed == ["reset", "reset_result", "step_chunk", "step_chunk"]
    assert (
        len(posted) == 5
    )  # Pending request replayed, physical operation executed once.
    assert posted[2] == posted[3]
    assert len(kills) == 2 and all(
        item == {"data": {"worker_ids": ["0"]}} for item in kills
    )
    assert len(agents[0].observations[0].media) == 2
