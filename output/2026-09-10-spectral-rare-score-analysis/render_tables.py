#!/usr/bin/env python3
"""Render existing metrics JSON only; never opens scientific arrays."""
import hashlib
import json
import os
from pathlib import Path
import resource
import signal

resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
signal.alarm(120)
here = Path(__file__).resolve().parent
data = (here / "metrics.json").read_bytes()
assert hashlib.sha256(data).hexdigest() == "f849489f3d96752033500b66e1606a89caa91552a5a08ebe34d872ed183ee7ec"
result = json.loads(data)
rows = {(r["seed"], r["cell"], r["policy"], r["step"], r["view"]): r["metrics"] for r in result["rows"]}
lines = ["# All-seed saved-logit results", "",
         "POST-HOC analytical transform, not an actual trained policy. Numerical output is complete,",
         "but the terminal resource-certification failure remains; see [provenance](provenance.json).",
         "Values before → after add log(11) to digit-8 logits. Accuracies and common→8 rates",
         "are percentages; CE uses natural logs. AUROC is identical before and after.", ""]
for cell in ("clean", "diffuse", "shared", "sham"):
    lines += ["## " + cell.title(), "",
              "| Seed | Policy | Rare AUROC | Rare accuracy | Common accuracy | Rare CE | Common CE | Common→8 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for seed in result["seed_order"]:
        for policy in ("raw", "native32", "norm_raw"):
            before = rows[(seed, cell, policy, 2000, "original")]
            after = rows[(seed, cell, policy, 2000, "plus_log11")]
            cells = [str(seed), policy, f'{before["rare_auroc"]:.6f}']
            for metric in ("rare_accuracy", "common_accuracy", "rare_ce", "common_ce", "common_predicted_8_rate"):
                scale = 1 if metric.endswith("_ce") else 100
                cells.append(f"{before[metric]*scale:.4f} → {after[metric]*scale:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
lines += ["## Common warmup, update 100", "",
          "Each seed's warmup is identical across all 12 logical branches.", "",
          "| Seed | Rare AUROC | Rare accuracy | Common accuracy | Rare CE | Common CE |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
for seed in result["seed_order"]:
    before = rows[(seed, "clean", "raw", 100, "original")]
    after = rows[(seed, "clean", "raw", 100, "plus_log11")]
    cells = [str(seed), f'{before["rare_auroc"]:.6f}']
    for metric in ("rare_accuracy", "common_accuracy", "rare_ce", "common_ce"):
        scale = 1 if metric.endswith("_ce") else 100
        cells.append(f"{before[metric]*scale:.4f} → {after[metric]*scale:.4f}")
    lines.append("| " + " | ".join(cells) + " |")
lines += ["", "All unrounded rows, seed summaries, sample SDs and paired contrasts: [metrics.json](metrics.json).", ""]
with (here / "all-seed-results.md").open("x") as handle:
    handle.write("\n".join(lines))
print("Rendered 36 endpoint pairs and 3 warmup pairs from existing metrics.json only.")
