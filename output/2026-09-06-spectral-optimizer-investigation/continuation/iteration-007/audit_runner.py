"""Inert audit entrypoint. Callable integration is registered CPU fixtures only."""
from __future__ import annotations


def run_audit_fixture(store, context, guard):
    import hashlib
    import os
    import torch
    import artifact_store as storage
    import audit_envelope as audit
    import identity_codec as codec
    import branch_execution as execution
    from scientific_runner import load_transaction, receipt_ref, utc_now

    if (context["profile"] != storage.MLP_FIXTURE or store.profile != storage.MLP_FIXTURE
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized()):
        raise RuntimeError("only hidden-CUDA MLP audit fixture is implemented")
    guard("audit.inputs.begin")
    inputs = load_transaction(store, context)
    name = codec.artifact_id(context["identity"], profile=context["profile"], kind="branch-results") + ".pt"
    report, _ = storage._inspect_dirfd(store._dirfd)
    row, = [row for row in report["receipts"] if row["name"] == name]
    receipt = {field: row[field] for field in ("schema", "name", "size", "sha256", "status", "encoding")}
    receipt["receipt_name"] = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
    branch = storage.ArtifactStore.load_tensor_tree(store.root, name,
        expected_size=receipt["size"], expected_sha256=receipt["sha256"])
    branch = execution._load_pinned(store, branch, receipt, schema_name="i7_branch_results")
    guard("audit.inputs.end")
    result = audit.audit_and_seal(store=store, branch=branch, branch_receipt=receipt,
        plan_root=store.root, plan_name="frozen-plan.pt", created_utc=utc_now(), guard=guard,
        **inputs, **context)
    guard("audit.seal.end")
    return [receipt_ref(result["receipt"])]


def main():
    print("No audit started. Scientific audit entrypoint remains disabled; use explicit fixture tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
