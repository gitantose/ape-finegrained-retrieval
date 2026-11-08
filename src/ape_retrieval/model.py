"""Pin artifacts and precision; reuse upstream loading and inference unchanged."""

import hashlib
import json
from pathlib import Path

import torch
from ape import APE, APEConfig
from huggingface_hub import snapshot_download


def configure_reference(config: dict, device: str) -> torch.device:
    if config["dtype"] != "float32" or config["tf32"] is not False:
        raise ValueError("The reference requires FP32 with TF32 disabled")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Choose an explicit cpu or cuda device")
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU fallback")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    return target


def load_model(config: dict, device: torch.device, download: bool = False) -> APE:
    cache_dir = Path(".cache/huggingface").resolve()
    model_path = Path(snapshot_download(
        config["model_id"], revision=config["model_revision"],
        allow_patterns=["config.json", "model.safetensors", "README.md"],
        cache_dir=cache_dir, local_files_only=not download,
    ))
    for filename, key in [("config.json", "config_sha256"),
                          ("model.safetensors", "model_sha256")]:
        with (model_path / filename).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != config[key]:
            raise ValueError(f"SHA-256 mismatch for {filename}: {actual}")

    # The full APE checkpoint includes its fine-tuned encoder weights.
    # Only the original backbone architecture and tokenizer are needed here.
    backbone_path = snapshot_download(
        config["backbone_id"], revision=config["backbone_revision"],
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "vocab.txt", "README.md"],
        cache_dir=cache_dir, local_files_only=not download,
    )
    architecture = json.loads((model_path / "config.json").read_text())
    architecture["encoder_name"] = str(Path(backbone_path).resolve())
    model = APE.from_pretrained(
        str(model_path), config=APEConfig(**architecture), local_files_only=True, strict=True,
    )
    return model.to(device=device, dtype=torch.float32).eval()
