"""Shared helpers for pipeline entrypoints constructing child CLI invocations."""


def parallel_incremental_filter_args(incremental_only: bool) -> list[str]:
    """Build argv fragment for parallel_af2rank_scoring / parallel_proteinebm_scoring.

    Those scripts only define ``--filter_existing`` as ``store_true`` (incremental:
    only proteins still needing work). When doing a full rerun, pass no flag so all
    CSV proteins are considered.
    """
    return ["--filter_existing"] if incremental_only else []
