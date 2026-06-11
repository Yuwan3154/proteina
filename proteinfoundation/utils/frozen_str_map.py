import pickle

import numpy as np


class FrozenStrMap:
    """Refcount-free, read-only ``{str: value}`` map for use inside forked DataLoader workers.

    A plain Python dict accessed every ``__getitem__`` grows host RAM in forked workers:
    CPython refcount/GC writes flip the dict's per-entry object pages, so copy-on-write
    replicates them per worker (pytorch#13246). This stores the whole map as THREE single
    C buffers -- a sorted numpy bytes key array, one concatenated pickle blob of the
    values, and an int64 offset array -- so a lookup touches no per-entry Python object's
    refcount and nothing is CoW-replicated. Drop-in for the read-only ``.get`` / ``in`` /
    ``len`` the dataset transforms use. Keys must be str/bytes; values may be any picklable
    object (lists, nested dicts, strings).
    """

    def __init__(self, d):
        items = sorted(
            ((k.encode() if isinstance(k, str) else bytes(k)), v) for k, v in d.items()
        )
        self._n = len(items)
        if self._n:
            self._keys = np.array([k for k, _ in items], dtype=np.bytes_)
            blobs = [pickle.dumps(v, protocol=pickle.HIGHEST_PROTOCOL) for _, v in items]
            self._off = np.zeros(self._n + 1, dtype=np.int64)
            self._off[1:] = np.cumsum(np.array([len(b) for b in blobs], dtype=np.int64))
            self._blob = np.frombuffer(b"".join(blobs), dtype=np.uint8)
        else:
            self._keys = np.array([], dtype="S1")
            self._off = np.zeros(1, dtype=np.int64)
            self._blob = np.array([], dtype=np.uint8)
        self._itemsize = int(self._keys.itemsize)

    def _find(self, key) -> int:
        kb = key.encode() if isinstance(key, str) else bytes(key)
        # len > itemsize cannot be a stored key, and numpy would silently TRUNCATE it.
        if self._n == 0 or len(kb) > self._itemsize:
            return -1
        i = int(np.searchsorted(self._keys, kb))
        if i < self._n and self._keys[i] == kb:
            return i
        return -1

    def get(self, key, default=None):
        i = self._find(key)
        if i < 0:
            return default
        return pickle.loads(self._blob[self._off[i] : self._off[i + 1]].tobytes())

    def __contains__(self, key) -> bool:
        return self._find(key) >= 0

    def __len__(self) -> int:
        return self._n
