"""Unit tests for proteinfoundation.utils.logtrace. Stdlib only -- no pytest.

Run: python tests/test_logtrace.py   (exits non-zero on any failure)

The property that actually matters is `on disk WITHOUT closing`: the DDP deadlock this tool
diagnoses KILLS the process, so a test that closes the handle first would pass even with a
fully-buffered handle and would prove nothing.
"""

import os
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinfoundation.utils import logtrace

_FAILURES = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def read_lines(d, rank):
    p = os.path.join(d, f"logtrace_rank{rank}.txt")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return fh.read().splitlines()


def case(fn):
    """Run one test with a fresh tmpdir and a clean env/handle cache."""
    name = fn.__name__
    d = tempfile.mkdtemp(prefix="logtrace_")
    prev = os.environ.get("LOGTRACE")
    logtrace.reset()
    os.environ.pop("LOGTRACE", None)
    try:
        fn(d)
        print(f"  PASS  {name}")
    except Exception as e:
        _FAILURES.append((name, e, traceback.format_exc()))
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    finally:
        logtrace.reset()
        os.environ.pop("LOGTRACE", None)
        if prev is not None:
            os.environ["LOGTRACE"] = prev
        shutil.rmtree(d, ignore_errors=True)


def t_disabled_writes_nothing(d):
    check(logtrace.trace(0, "train/loss", True, d) is False, "should return False when disabled")
    check(not os.path.exists(os.path.join(d, "logtrace_rank0.txt")), "no file may be created")


def t_enabled_format(d):
    os.environ["LOGTRACE"] = "1"
    check(logtrace.trace(0, "train/loss", True, d) is True, "should return True when enabled")
    check(read_lines(d, 0) == ["train/loss\tsync_dist=True"], f"bad content: {read_lines(d, 0)}")


def t_sync_dist_false(d):
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(0, "train/aux", False, d)
    check(read_lines(d, 0) == ["train/aux\tsync_dist=False"], f"bad: {read_lines(d, 0)}")


def t_one_line_per_call_ordered(d):
    os.environ["LOGTRACE"] = "1"
    for i in range(5):
        logtrace.trace(0, f"m{i}", i % 2 == 0, d)
    want = [f"m{i}\tsync_dist={i % 2 == 0}" for i in range(5)]
    check(read_lines(d, 0) == want, f"got {read_lines(d, 0)}")


def t_handle_reused(d):
    """The whole point of the change: ONE handle, not an open() per call."""
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(3, "a", True, d)
    first = logtrace._HANDLES[3]
    for _ in range(50):
        logtrace.trace(3, "a", True, d)
    check(logtrace._HANDLES[3] is first, "handle was reopened instead of reused")
    check(len(logtrace._HANDLES) == 1, f"unexpected handles: {list(logtrace._HANDLES)}")
    check(len(read_lines(d, 3)) == 51, f"expected 51 lines, got {len(read_lines(d, 3))}")


def t_on_disk_without_closing(d):
    """Survives a hard kill: an INDEPENDENT fd must see the line while the handle is still open."""
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(0, "before_crash", True, d)
    check(not logtrace._HANDLES[0].closed, "handle must still be open for this test to mean anything")
    with open(os.path.join(d, "logtrace_rank0.txt")) as fh:
        check(fh.read() == "before_crash\tsync_dist=True\n", "line was buffered, NOT flushed to disk")


def t_ranks_separate(d):
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(0, "r0", True, d)
    logtrace.trace(1, "r1", False, d)
    check(read_lines(d, 0) == ["r0\tsync_dist=True"], "rank0 wrong")
    check(read_lines(d, 1) == ["r1\tsync_dist=False"], "rank1 wrong")
    check(set(logtrace._HANDLES) == {0, 1}, "expected one handle per rank")


def t_append_not_truncate(d):
    """Mode must be 'a': a chain segment reopening must not wipe the previous segment."""
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(0, "seg1", True, d)
    logtrace.reset()
    logtrace.trace(0, "seg2", True, d)
    check(read_lines(d, 0) == ["seg1\tsync_dist=True", "seg2\tsync_dist=True"], f"got {read_lines(d, 0)}")


def t_reset_clears(d):
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(0, "x", True, d)
    logtrace.reset()
    check(logtrace._HANDLES == {}, "reset must clear the cache")


def t_creates_missing_dir(d):
    os.environ["LOGTRACE"] = "1"
    nested = os.path.join(d, "deep", "logs")
    check(logtrace.trace(0, "x", True, nested) is True, "should create the dir and write")
    check(os.path.exists(os.path.join(nested, "logtrace_rank0.txt")), "file missing")


def t_only_exactly_1_enables(d):
    for val in ("0", "true", "yes", "", "TRUE", "2"):
        os.environ["LOGTRACE"] = val
        check(logtrace.trace(0, "x", True, d) is False, f"LOGTRACE={val!r} must NOT enable")
    check(not os.path.exists(os.path.join(d, "logtrace_rank0.txt")), "no file may be created")


def t_reopens_if_externally_closed(d):
    os.environ["LOGTRACE"] = "1"
    logtrace.trace(0, "a", True, d)
    logtrace._HANDLES[0].close()          # simulate something closing it underneath us
    check(logtrace.trace(0, "b", True, d) is True, "must recover from a closed handle")
    check(read_lines(d, 0) == ["a\tsync_dist=True", "b\tsync_dist=True"], f"got {read_lines(d, 0)}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    print(f"running {len(tests)} logtrace tests")
    for fn in tests:
        case(fn)
    if _FAILURES:
        print(f"\n{len(_FAILURES)} FAILED:")
        for name, _, tb in _FAILURES:
            print(f"--- {name} ---\n{tb}")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")
