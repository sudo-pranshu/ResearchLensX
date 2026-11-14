"""Original-document storage behind a narrow interface.

The engine and the API only use `DocumentStorage`; `LocalDocumentStorage` keeps files on the local filesystem. A hosted
object store would be another implementation of the same Protocol (see docs/deployment-readiness/FUTURE_DEPLOYMENT_READINESS.md).
Every call takes the **workspace id** explicitly, so storage keys are always scoped.

Local layout: <root>/<workspace>/<safe doc id>/original.pdf. Writes are atomic (temp file + fsync + rename). Document ids are
mapped to filesystem-safe names (with a short hash to keep distinct ids distinct), so no id can escape the root.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class StoredFile:
    key: str            # opaque storage key; for the local adapter a path relative to the root
    sha256: str
    bytes: int


class DocumentStorage(Protocol):
    def put(self, workspace: str, doc_id: str, data: bytes) -> StoredFile: ...
    def open_read(self, workspace: str, doc_id: str) -> BinaryIO: ...
    def exists(self, workspace: str, doc_id: str) -> bool: ...
    def delete(self, workspace: str, doc_id: str) -> bool: ...
    def presigned_url(self, workspace: str, doc_id: str, expires_s: int = 300) -> str | None:
        """A time-limited URL for direct download, or None if the backend cannot provide one (the local adapter cannot)."""
        ...


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    if not value:
        raise ValueError("empty identifier")
    base = _SAFE.sub("_", value).strip("._") or "x"
    return f"{base[:64]}-{hashlib.sha1(value.encode()).hexdigest()[:8]}"


class LocalDocumentStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _dir(self, workspace: str, doc_id: str) -> Path:
        return self.root / _safe_name(workspace) / _safe_name(doc_id)

    def _file(self, workspace: str, doc_id: str) -> Path:
        return self._dir(workspace, doc_id) / "original.pdf"

    def put(self, workspace: str, doc_id: str, data: bytes) -> StoredFile:
        d = self._dir(workspace, doc_id)
        d.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, d / "original.pdf")              # atomic on the same filesystem
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return StoredFile(str(self._file(workspace, doc_id).relative_to(self.root)), hashlib.sha256(data).hexdigest(), len(data))

    def open_read(self, workspace: str, doc_id: str) -> BinaryIO:
        return open(self._file(workspace, doc_id), "rb")

    def exists(self, workspace: str, doc_id: str) -> bool:
        return self._file(workspace, doc_id).is_file()

    def delete(self, workspace: str, doc_id: str) -> bool:
        d = self._dir(workspace, doc_id)
        if not d.exists():
            return False
        shutil.rmtree(d)
        return True

    def presigned_url(self, workspace: str, doc_id: str, expires_s: int = 300) -> str | None:
        return None

    def path_for(self, workspace: str, doc_id: str) -> Path:
        """Local-only helper (not part of the Protocol): used by tests and the integrity check."""
        return self._file(workspace, doc_id)
