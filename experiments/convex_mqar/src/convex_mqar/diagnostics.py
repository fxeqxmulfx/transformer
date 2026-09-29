"""Read-only validation diagnosis; attention weights are reconstructed in FP32.

Local source: 2312.04927v1 app:synthetic. These observations do not identify a
causal effect of RoPE; that requires separately trained positional controls.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import Config
from .data import load_split
from .kernels import autocast
from .rope import RopeTransformer


def attention_weights(attention, features, positions, config):
    batch, length, width = features.shape
    with autocast(config):
        qkv = attention.qkv(features).reshape(batch, length, 3, attention.heads, attention.head_width)
        q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attention.rotate(q), attention.rotate(k)
    q = q.gather(2, positions[:, None, :, None].expand(-1, attention.heads, -1, attention.head_width))
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(width // attention.heads)
    future = torch.arange(length, device=features.device)[None, None, None, :] > positions[:, None, :, None]
    return scores.masked_fill(future, -torch.inf).softmax(-1).mean(1)


@torch.no_grad()
def diagnose_split(model, data, config, count):
    cached = {}
    handles = [block.attention.register_forward_pre_hook(
        lambda module, args, index=index: cached.__setitem__(index, args[0].detach()))
        for index, block in enumerate(model.blocks)]
    totals = dict(queries=0, correct=0, predicted_in_dictionary=0, restricted_correct=0,
                  loss=0., first_previous_mass=0., first_previous_top=0,
                  second_value_mass=0., second_value_top=0, second_key_mass=0.)
    gaps = {}
    try:
        for start in range(0, count, 128):
            tokens, positions, labels = (x[start:min(start+128, count)].to(config.device) for x in data)
            with autocast(config):
                logits = model(tokens, positions)
                loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
            pairs = positions.shape[1]
            keys, values = tokens[:, :2*pairs:2], tokens[:, 1:2*pairs:2]
            query_keys = tokens.gather(1, positions)
            slot = (query_keys[:, :, None] == keys[:, None, :]).long().argmax(-1)
            value_positions, key_positions = 2 * slot + 1, 2 * slot
            predictions = logits.argmax(-1)
            restricted = logits.gather(2, values[:, None, :].expand(-1, pairs, -1)).argmax(-1)
            restricted_predictions = values.gather(1, restricted)
            previous_positions = torch.arange(1, 2*pairs, 2, device=config.device).expand(len(tokens), -1)
            first = attention_weights(model.blocks[0].attention, cached[0], previous_positions, config)
            second = attention_weights(model.blocks[1].attention, cached[1], positions, config)
            correct = predictions == labels
            totals["queries"] += labels.numel()
            totals["correct"] += correct.sum().item()
            totals["predicted_in_dictionary"] += (predictions[:, :, None] == values[:, None, :]).any(-1).sum().item()
            totals["restricted_correct"] += (restricted_predictions == labels).sum().item()
            totals["loss"] += loss.item()
            totals["first_previous_mass"] += first.gather(2, (previous_positions-1)[..., None]).sum().item()
            totals["first_previous_top"] += (first.argmax(-1) == previous_positions-1).sum().item()
            totals["second_value_mass"] += second.gather(2, value_positions[..., None]).sum().item()
            totals["second_value_top"] += (second.argmax(-1) == value_positions).sum().item()
            totals["second_key_mass"] += second.gather(2, key_positions[..., None]).sum().item()
            distance = positions - value_positions
            for group in range(4):
                mask = (distance >= 16*group) & (distance < 16*(group+1))
                hit, total = gaps.get(group, (0, 0))
                gaps[group] = hit + (correct & mask).sum().item(), total + mask.sum().item()
    finally:
        for handle in handles:
            handle.remove()
    n = totals["queries"]
    return {"sequences": count, "queries": n, "accuracy": totals["correct"] / n,
            "loss": totals["loss"] / n,
            "prediction_is_a_prefix_value": totals["predicted_in_dictionary"] / n,
            "accuracy_with_oracle_candidate_values": totals["restricted_correct"] / n,
            "layer1_previous_key_mean_mass": totals["first_previous_mass"] / n,
            "layer1_previous_key_top_attention": totals["first_previous_top"] / n,
            "layer2_matching_value_mean_mass": totals["second_value_mass"] / n,
            "layer2_matching_value_top_attention": totals["second_value_top"] / n,
            "layer2_matching_key_mean_mass": totals["second_key_mass"] / n,
            "accuracy_by_value_query_distance": {f"{16*k}-{16*k+15}": {
                "queries": total, "accuracy": hit / total if total else None} for k, (hit, total) in gaps.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = Config(**json.loads((args.run / "config.json").read_text()))
    comparison = json.loads((args.run / "comparison.json").read_text())
    row = next(row for row in comparison["results"] if row["length"] == 64)
    selected = row["selected_run"]
    checkpoint = args.run / f"n64-d{row['width']}-lr{selected['learning_rate']:.8g}" / "best.pt"
    model = RopeTransformer(config.vocab, row["width"], config.layers, config.heads,
                            config.mlp_ratio, config.rope_base).to(config.device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=config.device, weights_only=True))
    report = {"length": 64, "width": row["width"], "learning_rate": selected["learning_rate"],
              "checkpoint": str(checkpoint), "splits": {},
              "scope": "Read-only train/validation diagnostics; no test-based selection or retraining",
              "candidate_control": "Oracle restricts readout candidates; not a standalone model",
              "causal_scope": "Association of errors with routing; no causal attribution to RoPE"}
    for split in ("train", "validation"):
        data, identity = load_split(args.data, config, 64, split)
        report["splits"][split] = diagnose_split(model, data, config, min(3000, len(data[0]))) | {"identity": identity}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
