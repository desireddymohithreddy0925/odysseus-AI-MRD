import json

import pytest

from src import tool_implementations as tools
from src.tools import cookbook


@pytest.mark.asyncio
async def test_launch_model_agent_dry_run_uses_official_hf_page_and_mlx_engine(monkeypatch):
    async def fake_servers():
        return {
            "default_host": "pewds@mac-studio",
            "hosts": [
                {
                    "name": "epictetus",
                    "host": "pewds@mac-studio",
                    "platform": "macos",
                    "env": "venv",
                    "envPath": "/Users/pewds/src/exo/.venv",
                }
            ],
        }

    async def fake_hf_info(repo_id):
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "tags": ["mlx"],
            "siblings": ["model.safetensors"],
        }

    monkeypatch.setattr(cookbook, "_cookbook_servers", fake_servers)
    monkeypatch.setattr(cookbook, "_cookbook_hf_model_info", fake_hf_info)

    result = await tools.do_launch_model_agent(
        json.dumps({"repo_id": "mlx-community/DeepSeek-V4-Flash-4bit", "dry_run": True})
    )

    assert result["exit_code"] == 0
    assert result["plan"]["official_url"] == "https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit"
    assert result["plan"]["engine"] == "mlx"
    assert result["plan"]["cmd"].startswith("python3 -m mlx_lm.server")
    assert "pewds@mac-studio" in result["output"]


@pytest.mark.asyncio
async def test_launch_model_agent_dry_run_defaults_to_vllm_for_hf_transformer(monkeypatch):
    async def fake_servers():
        return {"default_host": "", "hosts": []}

    async def fake_hf_info(repo_id):
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "tags": ["transformers"],
            "siblings": ["config.json", "model-00001-of-00002.safetensors"],
        }

    monkeypatch.setattr(cookbook, "_cookbook_servers", fake_servers)
    monkeypatch.setattr(cookbook, "_cookbook_hf_model_info", fake_hf_info)

    result = await tools.do_launch_model_agent(
        json.dumps({"repo_id": "Qwen/Qwen3-8B", "dry_run": True, "port": 8010})
    )

    assert result["exit_code"] == 0
    assert result["plan"]["engine"] == "vllm"
    assert result["plan"]["cmd"] == "vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8010"


@pytest.mark.asyncio
async def test_launch_model_agent_retries_structured_diagnosis(monkeypatch):
    async def fake_servers():
        return {"default_host": "", "hosts": []}

    async def fake_hf_info(repo_id):
        return {
            "repo_id": repo_id,
            "url": f"https://huggingface.co/{repo_id}",
            "tags": ["transformers"],
            "siblings": ["config.json"],
        }

    launched = []

    async def fake_serve_model(content, owner=None):
        args = json.loads(content)
        launched.append(args["cmd"])
        return {"exit_code": 0, "session_id": f"serve-{len(launched)}"}

    async def fake_list_served_models(content, owner=None):
        sid = f"serve-{len(launched)}"
        if len(launched) == 1:
            return {
                "exit_code": 0,
                "tasks": [
                    {
                        "session_id": sid,
                        "status": "error",
                        "diagnosis": {
                            "message": "GPU ran out of memory during startup or warmup.",
                            "suggestions": [
                                {
                                    "label": "retry with GPU memory utilization 0.80",
                                    "op": "replace",
                                    "flag": "--gpu-memory-utilization",
                                    "value": "0.80",
                                }
                            ],
                        },
                    }
                ],
            }
        return {"exit_code": 0, "tasks": [{"session_id": sid, "status": "ready"}]}

    async def fake_tail(content, owner=None):
        return {"exit_code": 0, "output": "torch.cuda.OutOfMemoryError"}

    monkeypatch.setattr(cookbook, "_cookbook_servers", fake_servers)
    monkeypatch.setattr(cookbook, "_cookbook_hf_model_info", fake_hf_info)
    monkeypatch.setattr(cookbook, "do_serve_model", fake_serve_model)
    monkeypatch.setattr(cookbook, "do_list_served_models", fake_list_served_models)
    monkeypatch.setattr(cookbook, "do_tail_serve_output", fake_tail)

    result = await tools.do_launch_model_agent(
        json.dumps(
            {
                "repo_id": "Qwen/Qwen3-8B",
                "cmd": "vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.95",
                "max_attempts": 2,
                "poll_attempts": 1,
            }
        )
    )

    assert result["exit_code"] == 0
    assert launched == [
        "vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.95",
        "vllm serve Qwen/Qwen3-8B --port 8000 --gpu-memory-utilization 0.80",
    ]
    assert result["session_id"] == "serve-2"
