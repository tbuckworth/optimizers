"""Read-only operator geometry of saved seed100 filter bases; no training."""
import argparse
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from experiments.grokking_confirmation import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Existing analysis output; do not overwrite.")
    torch.set_num_threads(1)
    completion = json.loads((args.batch_root / "complete.json").read_text())
    rows = []
    for entry in completion["accepted_results"]:
        if entry["seed"] != 100 or entry["arm"] == "adamw":
            continue
        for receipt in entry["checkpoint_receipts"]:
            path = Path(receipt["path"])
            step = int(path.stem.rsplit("-", 1)[1])
            if step not in (100, 1000, 2500, 4000, 6000):
                continue
            state = load_checkpoint(path, receipt["sha256"])
            basis = state["filter_state"]["V"]
            row = {"seed": 100, "arm": entry["arm"], "step": step,
                   "checkpoint": str(path), "checkpoint_sha256": receipt["sha256"],
                   "basis_rank": 0 if basis is None else basis.shape[1]}
            if basis is not None:
                double_basis = basis.double()
                gram = double_basis.T @ double_basis
                eigenvalues = torch.linalg.eigvalsh(gram)
                row.update({"minimum_nonzero_operator_gain": float(eigenvalues.min()),
                            "maximum_operator_gain": float(eigenvalues.max()),
                            "orthogonality_error_spectral": float((eigenvalues-1).abs().max()),
                            "orthogonality_error_frobenius": float(torch.linalg.vector_norm(eigenvalues-1))})
                del double_basis, gram, eigenvalues
            rows.append(row)
            del state, basis
    if len(rows) != 10:
        raise ValueError("Expected five fixed checkpoints for each filtered arm.")
    result = {"status": "read-only saved-basis geometry, not a causal intervention",
              "precision": "fp64 Gram of the exact saved fp32 basis entries, CPU one thread",
              "interpretation": "nonzero eigenvalues of V V^T equal eigenvalues of V^T V; unit gains are an orthogonal projector",
              "rows": rows}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
