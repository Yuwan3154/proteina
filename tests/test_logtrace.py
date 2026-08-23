"""Unit tests for proteinfoundation.utils.logtrace.

The property that actually matters: every completed line must be on disk WITHOUT the handle
being closed, because the DDP deadlock this tool diagnoses kills the process. A test that
closes the file first would pass even with a fully-buffered handle and prove nothing.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.utils import logtrace


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    logtrace.reset()
    monkeypatch.delenv("LOGTRACE", raising=False)
    yield
    logtrace.reset()


def _read(tmp_path, rank):
    p = tmp_path / f"logtrace_rank{rank}.txt"
    return p.read_text().splitlines() if p.exists() else None


def test_disabled_writes_nothing_and_creates_no_file(tmp_path):
    assert logtrace.trace(0, "train/loss", True, str(tmp_path)) is False
    assert not (tmp_path / "logtrace_rank0.txt").exists()


def test_enabled_writes_expected_format(monkeypatch, tmp_path):
    monkeypatch.setenv("LOGTRACE", "1")
    assert logtrace.trace(0, "train/loss", True, str(tmp_path)) is True
    assert _read(tmp_path, 0) == ["train/loss\tsync_dist=True"]


def test_sync_dist_false_recorded(monkeypatch, tmp_path):
    monkeypatch.setenv("LOGTRACE", "1")
    logtrace.trace(0, "train/aux", False, str(tmp_path))
    assert _read(tmp_path, 0) == ["train/aux\tsync_dist=False"]


def test_one_line_per_call_in_order(monkeypatch, tmp_path):
    monkeypatch.setenv("LOGTRACE", "1")
    for i in range(5):
        logtrace.trace(0, f"m{i}", i % 2 == 0, str(tmp_path))
    assert _read(tmp_path, 0) == [
        "m0\tsync_dist=True", "m1\tsync_dist=False", "m2\tsync_dist=True",
        "m3\tsync_dist=False", "m4\tsync_dist=True",
    ]


def test_handle_is_reused_not_reopened(monkeypatch, tmp_path):
    """The whole point of the change: one handle, not an open() per call."""
    monkeypatch.setenv("LOGTRACE", "1")
    logtrace.trace(3, "a", True, str(tmp_path))
    first = logtrace._HANDLES[3]
    for _ in range(50):
        logtrace.trace(3, "a", True, str(tmp_path))
    assert logtrace._HANDLES[3] is first
    assert len(logtrace._HANDLES) == 1


def test_lines_are_on_disk_without_closing(monkeypatch, tmp_path):
    """Survives a hard kill: readable by an INDEPENDENT open() while the handle is still live."""
    monkeypatch.setenv("LOGTRACE", "1")
    logtrace.trace(0, "before_crash", True, str(tmp_path))
    assert not logtrace._HANDLES[0].closed          # deliberately NOT closed
    with open(tmp_path / "logtrace_rank0.txt") as fh:  # separate fd, sees only flushed bytes
        assert fh.read() == "before_crash\tsync_dist=True\n"


def test_ranks_get_separate_files(monkeypatch, tmp_path):
    monkeypatch.setenv("LOGTRACE", "1")
    logtrace.trace(0, "r0", True, str(tmp_path))
    logtrace.trace(1, "r1", False, str(tmp_path))
    assert _read(tmp_path, 0) == ["r0\tsync_dist=True"]
    assert _read(tmp_path, 1) == ["r1\tsync_dist=False"]
    assert set(logtrace._HANDLES) == {0, 1}


def test_reopens_after_reset(monkeypatch, tmp_path):
    monkeypatch.setenv("LOGTRACE", "1")
    logtrace.trace(0, "one", True, str(tmp_path))
    logtrace.reset()
    assert logtrace._HANDLES == {}
    logtrace.trace(0, "two", True, str(tmp_path))
    assert _read(tmp_path, 0) == ["one\tsync_dist=True", "two\tsync_dist=True"]


def test_appends_across_reset_does_not_truncate(monkeypatch, tmp_path):
    """Mode must be 'a': a chain segment reopening must not wipe the previous segment."""
    monkeypatch.setenv("LOGTRACE", "1")
    logtrace.trace(0, "seg1", True, str(tmp_path))
    logtrace.reset()
    logtrace.trace(0, "seg2", True, str(tmp_path))
    assert len(_read(tmp_path, 0)) == 2


def test_creates_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("LOGTRACE", "1")
    nested = tmp_path / "deep" / "logs"
    assert logtrace.trace(0, "x", True, str(nested)) is True
    assert (nested / "logtrace_rank0.txt").exists()


def test_env_other_than_1_is_disabled(monkeypatch, tmp_path):
    for val in ("0", "true", "yes", ""):
        monkeypatch.setenv("LOGTRACE", val)
        assert logtrace.trace(0, "x", True, str(tmp_path)) is False
    assert not (tmp_path / "logtrace_rank0.txt").exists()
