"""Shared synthetic CPU-only fixtures, never scientific provenance or input."""
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import tempfile

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("explicitly hide CUDA for envelope fixtures")

import torch
import torch.nn.functional as F

import artifact_store as storage
import data_probe_bindings as data
import source_environment as provenance
import state_core as state
import verified_plan_load as verified
from test_bound_data_pipeline import synthetic_idx
from test_plan_bindings import fixture as plan_fixture
from test_full_measurement_pipeline import model_factory, optimizer_factory, observer_factory


@contextmanager
def fixture_context(*, plan_handle=None, store=None):
    """Real temporary clean Git + runtime snapshot + sealed plan + synthetic IDX.

    The two-file repository is explicitly the collector's fixture profile, not
    evidence that these test modules were imported from it. Every temporary file
    is scoped to this context and no scientific source collector is invoked.
    """
    saved_rng = state._raw_rng_state()
    try:
        with tempfile.TemporaryDirectory(prefix="i7-envelope-fixture-") as directory:
            repo = Path(directory, "repo")
            repo.mkdir()
            (repo / "fixture_source.py").write_bytes(b"VALUE = 1\n")
            (repo / "fixture_contract.md").write_bytes(b"Synthetic collector fixture only.\n")
            for arguments in (("init", "-q"), ("add", "--", "fixture_source.py", "fixture_contract.md"),
                              ("-c", "user.name=Codex", "-c", "user.email=codex@example.invalid",
                               "-c", "commit.gpgsign=false", "commit", "-q", "-m", "Freeze synthetic inputs")):
                subprocess.run(["git", *arguments], cwd=repo, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            sources = provenance.collect_verified_sources(repo, profile=storage.MLP_FIXTURE)
            environment = provenance.collect_runtime_environment(repo, profile=storage.MLP_FIXTURE,
                                                                  runtime_role="fixture_cpu")
            identity, plan, _ = plan_fixture()
            if store is None:
                with storage.ArtifactStore(directory, profile=storage.MLP_FIXTURE,
                                           min_filesystem_free_bytes=0) as plan_store:
                    receipt = plan_store.write_tensor_tree("frozen-plan.pt", plan)
                loaded = verified.load_verified_plan(plan_store.root, "frozen-plan.pt", identity=identity,
                    profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
            else:
                if type(store) is not storage.ArtifactStore or store.profile != storage.MLP_FIXTURE:
                    raise ValueError("shared fixture store must have exact MLP profile")
                plan_store = store
                receipt = store.write_tensor_tree("frozen-plan.pt", plan)
                loaded = verified.load_verified_plan_from_store(store, "frozen-plan.pt", identity=identity,
                    profile=storage.MLP_FIXTURE, expected_sha256=receipt["sha256"])
            if plan_handle is not None:
                if type(plan_handle) is not dict or plan_handle:
                    raise ValueError("plan_handle must be an empty dict")
                plan_handle.update(root=plan_store.root, name="frozen-plan.pt", reference=loaded["artifact"])
            images, labels, expected_files = synthetic_idx()
            yield dict(identity=identity, profile=storage.MLP_FIXTURE, plan=loaded["plan"],
                       plan_artifact={"sha256":receipt["sha256"], "size_bytes":receipt["size"]},
                       images_bytes=images, labels_bytes=labels, expected_files=expected_files,
                       sources=sources, environment=environment)
    finally:
        state._set_rng_state(saved_rng)


def warm_live(context):
    """Four actual planned warm updates; leave the same live objects pre-forward."""
    identity, plan, profile = context["identity"], context["plan"], context["profile"]
    materialized = data.materialize(context["images_bytes"], context["labels_bytes"],
        plan=plan, identity=identity, profile=profile, expected_files=context["expected_files"])
    torch.manual_seed(plan["initialization_seed"])
    model = model_factory()
    optimizer = optimizer_factory(model)
    observer = observer_factory(model, optimizer)
    for local in plan["training_batches"][:identity["anchor_completed_updates"]]:
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(materialized["datasets"]["train_inputs"][local]),
                        materialized["datasets"]["train_noisy_labels"][local]).backward()
        observer.filter_grad()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer, observer
