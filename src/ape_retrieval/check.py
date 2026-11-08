"""Focused real-checkpoint checks, independent of benchmark data."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import time

import torch
from ape import APEScore

from ape_retrieval.model import configure_reference, load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--config", type=Path, default=Path("configs/reference.json"))
    parser.add_argument("--download", action="store_true", help="Allow pinned artifact acquisition")
    parser.add_argument("--output", type=Path,
                        help="Report path (default: runs/m1/<device>.json)")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("runs/m1") / f"{args.device}.json"
    config = json.loads(args.config.read_text())
    device = configure_reference(config, args.device)
    started = time.perf_counter()
    model = load_model(config, device, download=args.download)
    assert all(p.dtype == torch.float32 and p.device.type == device.type
               for p in model.parameters())
    sentences = [
        "Rome was built in 753 BC and is now the capital of Italy.",
        "Rome is the capital of Italy.",
        "",
    ]
    options = dict(batch_size=config["batch_size"], max_seq_len=config["max_seq_len"],
                   normalize_embeddings=config["normalize_embeddings"])
    encoding_started = time.perf_counter()
    embeddings, activities = model.encode(
        sentences, activity_threshold=config["activity_threshold"],
        return_scores=True, **options,
    )
    encoding_seconds = time.perf_counter() - encoding_started
    counts = [len(e) for e in embeddings]
    assert counts[:2] == [2, 1], counts
    for emb in embeddings:
        assert emb.dtype == torch.float32
        assert 1 <= len(emb) <= model.config.num_slots and emb.shape[1] == 384
        assert torch.isfinite(emb).all()
        torch.testing.assert_close(emb.norm(dim=1), torch.ones(len(emb)), atol=1e-6, rtol=1e-6)
    assert model.encode([], **options) == []
    # Upstream length sorting must restore input order across batches.
    for sentence, batched in zip(sentences, embeddings):
        single = model.encode(sentence, activity_threshold=config["activity_threshold"], **options)
        torch.testing.assert_close(single, batched, atol=2e-5, rtol=2e-5)

    all_slots, all_activities = model.encode(
        sentences, activity_threshold=-1., return_scores=True, **options,
    )
    fallback, fallback_activities = model.encode(
        sentences, activity_threshold=1., return_scores=True, **options,
    )
    raw_slots = model.encode(sentences, activity_threshold=-1.,
                             **{**options, "normalize_embeddings": False})
    for full, probs, selected, active_probs, strongest, strongest_prob, raw in zip(
            all_slots, all_activities, embeddings, activities, fallback,
            fallback_activities, raw_slots):
        assert torch.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()
        torch.testing.assert_close(full, torch.nn.functional.normalize(raw, dim=-1))
        mask = probs > config["activity_threshold"]
        if not mask.any():
            mask[probs.argmax()] = True
        torch.testing.assert_close(selected, full[mask])
        torch.testing.assert_close(active_probs, probs[mask])
        torch.testing.assert_close(strongest, full[probs.argmax()][None])
        torch.testing.assert_close(strongest_prob, probs.max()[None])

    # Distinct tails beyond the documented token limit must be truncated identically.
    prefix = "Rome is the capital of Italy. " * 40
    long_texts = [prefix + "The ending is blue.", prefix + "The ending is red."]
    token_lengths = [len(model.tokenizer(s)["input_ids"]) for s in long_texts]
    assert min(token_lengths) > config["max_seq_len"]
    truncated = model.encode(long_texts, **options)
    torch.testing.assert_close(truncated[0], truncated[1])
    query_first = APEScore(embeddings[1], embeddings[0]).h_min_to_p()
    reverse = APEScore(embeddings[0], embeddings[1]).h_min_to_p()
    assert query_first > reverse

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.with_suffix(".pt")
    torch.save({"embeddings": embeddings, "activities": activities}, tensor_path)
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path("src/ape_retrieval").glob("*.py"))}
    report = {
        "checks": "passed", "device": str(device), "config": config,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "ape_install": json.loads(importlib.metadata.distribution("ape").read_text("direct_url.json")),
        "source_sha256": source_hashes,
        "git_head": subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                                   capture_output=True, text=True).stdout.strip() or None,
        "sentences": sentences, "slot_counts": counts,
        "activities": [p.tolist() for p in activities],
        "all_activities": [p.tolist() for p in all_activities],
        "h2p_query_first": query_first, "h2p_reverse": reverse,
        "truncation_inputs": long_texts, "untruncated_token_lengths": token_lengths,
        "tensor_file": str(tensor_path),
        "tensor_sha256": hashlib.sha256(tensor_path.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "fixture_encoding_seconds": encoding_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "torch_cuda_runtime": torch.version.cuda,
    }
    if device.type == "cuda":
        report["gpu"] = torch.cuda.get_device_name(device)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(device).total_memory
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks": "passed", "device": str(device),
                      "slot_counts": counts, "report": str(args.output)}))


if __name__ == "__main__":
    main()
