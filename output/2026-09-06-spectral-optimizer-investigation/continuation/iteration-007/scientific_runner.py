"""Inert entrypoint; only explicit dataset-free fixture execution is implemented.

The name reserves the frozen scientific entrypoint. Native execution remains
disabled pending full resource accounting, producer integration and pilot GO.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def receipt_ref(receipt):
    return dict(name=receipt["name"], size_bytes=receipt["size"], sha256=receipt["sha256"])


def run_source_fixture(store, context, guard):
    """Eight real updates; retain the original live fifth step and final core."""
    import torch
    import torch.nn.functional as F
    import data_probe_bindings as data
    import source_capture as capture
    import state_core as state
    from branch_execution import _fixed_factories
    from phase_controller import SOURCE_COMPLETE

    profile, plan = context["profile"], context["plan"]
    if profile != "fixture_tiny_mlp_cpu_v1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("source runner is hidden-CUDA MLP fixture only")
    guard("source.inputs")
    materialized = data.materialize(context["images_bytes"], context["labels_bytes"], plan=plan,
        identity=context["identity"], profile=profile, expected_files=context["expected_files"])
    torch.manual_seed(plan["initialization_seed"])
    model_factory, optimizer_factory, observer_factory = _fixed_factories(profile, "cpu")
    model = model_factory()
    optimizer = optimizer_factory(model)
    observer = observer_factory(model, optimizer)
    transaction = None
    for update, indices in enumerate(plan["training_batches"], 1):
        guard(f"source.step.{update}.begin")
        optimizer.zero_grad(set_to_none=True)
        if update == 5:
            transaction = capture.capture_anchor_then_live_witness(model, optimizer, observer,
                store=store, created_utc=utc_now(), guard=guard, **context)
        else:
            loss = F.cross_entropy(model(materialized["datasets"]["train_inputs"][indices]),
                                   materialized["datasets"]["train_noisy_labels"][indices])
            guard(f"source.step.{update}.forward")
            loss.backward()
            guard(f"source.step.{update}.backward")
            observer.filter_grad()
            guard(f"source.step.{update}.observe")
            optimizer.step()
        guard(f"source.step.{update}.end")
    optimizer.zero_grad(set_to_none=True)
    endpoint = state.capture_core(model, optimizer, observer, profile=profile, completed_updates=8)
    guard("source.completion.pre_seal")
    receipt = store.write_tensor_tree(SOURCE_COMPLETE, endpoint)
    guard("source.completion.post_seal")
    return [receipt_ref(transaction["anchor_receipt"]), receipt_ref(transaction["witness_receipt"]),
            receipt_ref(receipt)]


def load_transaction(store, context):
    """Recover actual sealed producer inputs, not regenerated source witnesses."""
    import artifact_store as storage
    import identity_codec as codec
    import branch_execution as execution

    report, _ = storage._inspect_dirfd(store._dirfd)
    result = {}
    for kind, key, schema in (("anchor", "anchor", "i7_anchor"),
                              ("source-witness", "witness", "i7_source_step_witness")):
        name = codec.artifact_id(context["identity"], profile=context["profile"], kind=kind) + ".pt"
        rows = [row for row in report["receipts"] if row["name"] == name]
        if len(rows) != 1:
            raise RuntimeError("missing source input")
        base = rows[0]
        import hashlib
        receipt = {field: base[field] for field in ("schema", "name", "size", "sha256", "status", "encoding")}
        receipt["receipt_name"] = "receipt-" + hashlib.sha256(name.encode("ascii")).hexdigest() + ".json"
        value = storage.ArtifactStore.load_tensor_tree(store.root, name,
            expected_size=receipt["size"], expected_sha256=receipt["sha256"])
        result[key] = execution._load_pinned(store, value, receipt, schema_name=schema)
        result[key + "_receipt"] = receipt
    return result


def run_branches_fixture(store, context, guard):
    import branch_execution as execution
    transaction = load_transaction(store, context)
    result = execution.execute_branches(store=store, created_utc=utc_now(), guard=guard,
                                         **transaction, **context)
    return [receipt_ref(result["receipt"])]


def run_fixture(parent):
    """Explicit synthetic smoke run, retaining exactly one artifact budget root."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("explicitly hide CUDA for fixture execution")
    import torch
    import artifact_store as storage
    from envelope_fixture import fixture_context
    from phase_controller import PhaseController
    from audit_runner import run_audit_fixture

    if torch.cuda.is_initialized():
        raise RuntimeError("fixture requires uninitialized CUDA")
    store = storage.ArtifactStore(parent, profile=storage.MLP_FIXTURE, min_filesystem_free_bytes=0)
    controller = None
    try:
        with fixture_context(store=store) as context:
            controller = PhaseController(store, context=context, plan_name="frozen-plan.pt", created_utc=utc_now())
            # Exercise real persisted ready boundary without creating a second root.
            pin = controller.boundary_pin()
            store.close()
            controller = PhaseController.reopen(pin, context=context)
            store = controller.store
            controller.go("primary", decision="dataset_free_fixture_only", created_utc=utc_now())
            controller.execute("source", lambda held, guard: run_source_fixture(held, context, guard), created_utc=utc_now())
            controller.execute("branches", lambda held, guard: run_branches_fixture(held, context, guard), created_utc=utc_now())
            primary = controller.summary()
            pin = controller.boundary_pin()
            store.close()
            controller = PhaseController.reopen(pin, context=context)
            store = controller.store
            controller.go("audit", decision="dataset_free_fixture_only", created_utc=utc_now())
            controller.execute("audit", lambda held, guard: run_audit_fixture(held, context, guard), created_utc=utc_now())
            pin = controller.boundary_pin()
            store.close()
            controller = PhaseController.reopen(pin, context=context)
            store = controller.store
            return dict(primary=primary, final=controller.summary(), boundary_pin=pin,
                        scope="one synthetic CPU primary trajectory; no scientific execution")
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-parent", help="explicit existing directory for one synthetic CPU artifact root")
    arguments = parser.parse_args(argv)
    if arguments.fixture_parent is None:
        print("No execution requested. Scientific execution is disabled pending resource and pilot review.")
        return 0
    print(json.dumps(run_fixture(arguments.fixture_parent), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
