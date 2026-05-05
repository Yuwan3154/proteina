from pathlib import Path

from proteinfoundation.af2rank_evaluation.sharding_utils import (
    filter_proteins_threaded,
    shard_proteins,
    step_sentinel_path,
    wait_for_step,
)
from proteinfoundation.prediction_pipeline.run_prediction_pipeline import _cleanup_shard_sentinels


def test_filter_proteins_threaded_correctness():
    protein_ids = [f"p{i}" for i in range(8)]
    complete = {f"p{i}" for i in range(8) if i % 2 == 1}

    needing, done = filter_proteins_threaded(
        protein_ids,
        lambda protein_id: protein_id in complete,
        max_workers=3,
    )

    assert needing == ["p0", "p2", "p4", "p6"]
    assert done == ["p1", "p3", "p5", "p7"]


def test_filter_proteins_threaded_exception_safety():
    """Proteins whose check raises must end up in needing_work, not crash the pool."""
    protein_ids = ["ok_done", "ok_pending", "will_raise", "also_raise"]

    def _check(pid: str) -> bool:
        if pid.startswith("will_raise") or pid.startswith("also_raise"):
            raise RuntimeError(f"simulated I/O error for {pid}")
        return pid == "ok_done"

    needing, done = filter_proteins_threaded(protein_ids, _check, max_workers=2)

    assert "ok_done" in done
    assert "ok_pending" in needing
    # Proteins that raised must be treated as needing work (conservative).
    assert "will_raise" in needing
    assert "also_raise" in needing


def test_dynamic_slice_is_disjoint_partition():
    protein_ids = [f"p{i}" for i in range(13)]
    filtered = [protein_id for protein_id in protein_ids if protein_id not in {"p1", "p7", "p12"}]
    lengths = {protein_id: idx + 1 for idx, protein_id in enumerate(protein_ids)}
    shards = [shard_proteins(filtered, shard_idx, 4, lengths=lengths) for shard_idx in range(4)]

    seen = set()
    for shard in shards:
        shard_set = set(shard)
        assert not (seen & shard_set)
        seen.update(shard_set)
    assert seen == set(filtered)


def test_static_slice_preserved_for_tar():
    protein_ids = [f"p{i}" for i in range(12)]
    lengths = {protein_id: idx + 10 for idx, protein_id in enumerate(protein_ids)}
    static_shards = [shard_proteins(protein_ids, shard_idx, 3, lengths=lengths) for shard_idx in range(3)]

    remaining_after_step_a = [protein_id for protein_id in protein_ids if protein_id not in {"p0", "p3"}]
    remaining_after_step_b = [protein_id for protein_id in remaining_after_step_a if protein_id not in {"p5"}]

    assert [shard_proteins(protein_ids, shard_idx, 3, lengths=lengths) for shard_idx in range(3)] == static_shards
    dynamic_a = [shard_proteins(remaining_after_step_a, shard_idx, 3, lengths=lengths) for shard_idx in range(3)]
    dynamic_b = [shard_proteins(remaining_after_step_b, shard_idx, 3, lengths=lengths) for shard_idx in range(3)]
    assert dynamic_a != static_shards
    assert dynamic_b != static_shards
    assert set().union(*[set(shard) for shard in static_shards]) == set(protein_ids)


def test_step_sentinel_round_trip_and_cleanup(tmp_path: Path):
    step_sentinel_path(tmp_path, "inference", 0, 2).write_text("0")
    assert wait_for_step(tmp_path, "inference", 2, 1, True, poll_interval=0, timeout=1)

    step_sentinel_path(tmp_path, "proteinebm", 0, 2).write_text("1")
    assert not wait_for_step(tmp_path, "proteinebm", 2, 1, True, poll_interval=0, timeout=1)

    (tmp_path / ".shard_0_of_2_complete").write_text("0")
    (tmp_path / ".shard_1_of_2_complete").write_text("0")
    assert list(tmp_path.glob(".step_*_shard_*_of_*_complete"))
    _cleanup_shard_sentinels(tmp_path, 2)
    assert not list(tmp_path.glob(".step_*_shard_*_of_*_complete"))
    assert not list(tmp_path.glob(".shard_*_of_*_complete"))
