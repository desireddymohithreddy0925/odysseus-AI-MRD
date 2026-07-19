from services.hwfit import image_models

rank_image_models = image_models.rank_image_models
IMAGE_MODEL_REGISTRY = image_models.IMAGE_MODEL_REGISTRY

SYS = {"gpu_vram_gb": 0, "has_gpu": False}


def _disable_hf_discovery(monkeypatch):
    monkeypatch.setattr(image_models, "_fetch_hf_image_collection_models", lambda: [])
    monkeypatch.setattr(image_models, "_discover_quant_repos", lambda *a, **k: {})


def test_rank_image_models_handles_non_string_search(monkeypatch):
    _disable_hf_discovery(monkeypatch)
    # search is a CLI/API filter arg; a non-string made search.lower() raise
    # AttributeError. A non-string search should behave as "no filter".
    out = rank_image_models(SYS, search=123)
    assert len(out) == len(IMAGE_MODEL_REGISTRY)


def test_rank_image_models_string_filter_still_applies(monkeypatch):
    _disable_hf_discovery(monkeypatch)
    out = rank_image_models(SYS, search="zzzznotarealmodelzzz")
    assert out == []


def test_rank_image_models_uses_ram_budget_when_gpu_disabled(monkeypatch):
    _disable_hf_discovery(monkeypatch)

    gpu_out = rank_image_models({"has_gpu": True, "gpu_vram_gb": 8, "available_ram_gb": 64}, search="Qwen Image")
    ram_out = rank_image_models({"has_gpu": False, "gpu_vram_gb": 0, "available_ram_gb": 64}, search="Qwen Image")

    gpu_qwen = next(m for m in gpu_out if m["id"] == "Qwen/Qwen-Image")
    ram_qwen = next(m for m in ram_out if m["id"] == "Qwen/Qwen-Image")

    assert gpu_qwen["fit"] == "no_fit"
    assert gpu_qwen["quant"] == "FP8"
    assert gpu_qwen["fit_budget"] == "gpu"
    assert ram_qwen["fit"] in {"good", "perfect"}
    assert ram_qwen["quant"] == "BF16"
    assert ram_qwen["fit_budget"] == "ram"


def test_mlx_image_collection_models_only_show_on_apple(monkeypatch):
    mlx_model = {
        "id": "mlx-community/FLUX.2-klein-4B-bf16",
        "name": "FLUX.2 klein 4B bf16",
        "provider": "mlx-community",
        "params_b": 4.0,
        "vram_bf16": 10.0,
        "vram_fp8": None,
        "vram_q4": None,
        "default_quant": "BF16",
        "quant_repos": {},
        "capabilities": ["text-to-image"],
        "description": "Apple Silicon / MLX only.",
        "quality": 82,
        "speed": 88,
        "mlx_only": True,
    }
    monkeypatch.setattr(image_models, "_fetch_hf_image_collection_models", lambda: [mlx_model])
    monkeypatch.setattr(image_models, "_discover_quant_repos", lambda *a, **k: {})

    cuda = rank_image_models({"has_gpu": True, "gpu_vram_gb": 48, "backend": "cuda"}, search="FLUX.2 klein")
    metal = rank_image_models(
        {"has_gpu": True, "gpu_vram_gb": 48, "backend": "metal", "unified_memory": True},
        search="FLUX.2 klein",
    )

    assert cuda == []
    assert [m["id"] for m in metal] == ["mlx-community/FLUX.2-klein-4B-bf16"]


def test_apple_image_mode_hides_non_mlx_models(monkeypatch):
    _disable_hf_discovery(monkeypatch)

    metal = rank_image_models(
        {"has_gpu": True, "gpu_vram_gb": 48, "backend": "metal", "unified_memory": True},
        search="Qwen Image",
    )
    cuda = rank_image_models(
        {"has_gpu": True, "gpu_vram_gb": 48, "backend": "cuda"},
        search="Qwen Image",
    )

    assert metal == []
    assert any(m["id"] == "Qwen/Qwen-Image" for m in cuda)


def test_top_apple_image_seeds_show_on_metal(monkeypatch):
    _disable_hf_discovery(monkeypatch)

    seeds = [
        image_models._collection_item_to_model({"id": repo}, "Pinned Apple image models", mlx_only=True)
        for repo in image_models.HF_MLX_IMAGE_REPO_SEEDS
    ]
    monkeypatch.setattr(image_models, "_fetch_hf_image_collection_models", lambda: [s for s in seeds if s])

    metal = rank_image_models(
        {"has_gpu": True, "gpu_vram_gb": 64, "backend": "metal", "unified_memory": True},
        search="",
    )
    cuda = rank_image_models(
        {"has_gpu": True, "gpu_vram_gb": 64, "backend": "cuda"},
        search="",
    )
    metal_ids = {m["id"] for m in metal}
    cuda_ids = {m["id"] for m in cuda}

    assert set(image_models.HF_MLX_IMAGE_REPO_SEEDS).issubset(metal_ids)
    assert set(image_models.HF_MLX_IMAGE_REPO_SEEDS).isdisjoint(cuda_ids)
