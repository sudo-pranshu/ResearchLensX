"""Process-level configuration, read from the environment (and a `.env` file loaded by the launcher).

Request-level options (retriever, k, rerank...) live in `researchlens.workspace.Settings`; this module only holds what
belongs to the *process*: where data lives, limits, bind address. No path, host or limit is hard-coded elsewhere.

  RLX_DATA_DIR           root for local data (default: <repo>/data)
  RLX_DB                 SQLite library file (default: <data>/processed/library.sqlite)
  RLX_FILES_DIR          original PDFs (default: <data>/library)
  RLX_EMBEDDING_DIR      embedding stores (default: <repo>/artifacts/embeddings)
  RLX_HOST / RLX_PORT    bind address and preferred port (default 127.0.0.1 / 8501)
  RLX_MAX_UPLOAD_MB      per-file upload limit (default 50)
  RLX_MAX_PAGES          per-document page limit (default 400)
  RLX_DEVICE             auto | cpu | mps | cuda
  RLX_WORKSPACE          workspace id (default "local"; all stored data is scoped by it)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICES = ("auto", "cpu", "mps", "cuda")


class AppConfig(BaseModel):
    data_dir: Path
    db_path: Path
    files_dir: Path
    embedding_dir: Path
    host: str = "127.0.0.1"
    port: int = Field(8501, ge=1, le=65535)
    max_upload_mb: int = Field(50, ge=1, le=2048)
    max_pages: int = Field(400, ge=1, le=10000)
    device: str = "auto"
    workspace: str = Field("local", pattern=r"^[A-Za-z0-9_-]{1,64}$")

    @field_validator("device")
    @classmethod
    def _device(cls, v: str) -> str:
        if v not in DEVICES:
            raise ValueError(f"RLX_DEVICE must be one of {DEVICES}, got {v!r}")
        return v

    @field_validator("host")
    @classmethod
    def _host(cls, v: str) -> str:
        # There is no authentication yet: refuse to listen on all interfaces (see docs/architecture/LOCAL_ARCHITECTURE.md §11).
        if v in ("0.0.0.0", "::", ""):
            raise ValueError("RLX_HOST must not be a wildcard address: the local API has no authentication")
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, root: Path | None = None) -> "AppConfig":
        e = os.environ if env is None else env
        root = Path(root) if root else REPO_ROOT

        def path(name: str, default: Path) -> Path:
            v = e.get(name)
            if not v:
                return default
            p = Path(v).expanduser()
            return p if p.is_absolute() else root / p

        data = path("RLX_DATA_DIR", root / "data")
        try:
            return cls(
                data_dir=data,
                db_path=path("RLX_DB", data / "processed" / "library.sqlite"),
                files_dir=path("RLX_FILES_DIR", data / "library"),
                embedding_dir=path("RLX_EMBEDDING_DIR", root / "artifacts" / "embeddings"),
                host=e.get("RLX_HOST", "127.0.0.1"),
                port=int(e.get("RLX_PORT", 8501)),
                max_upload_mb=int(e.get("RLX_MAX_UPLOAD_MB", 50)),
                max_pages=int(e.get("RLX_MAX_PAGES", 400)),
                device=e.get("RLX_DEVICE", "auto"),
                workspace=e.get("RLX_WORKSPACE", "local"),
            )
        except ValueError as exc:      # int() failures and pydantic validation errors alike
            raise ValueError(f"invalid ResearchLens X configuration: {exc}") from exc
