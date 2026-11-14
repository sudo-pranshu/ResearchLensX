import pytest

from researchlens.settings import AppConfig


def test_defaults_are_under_the_repo_and_local_only(tmp_path):
    c = AppConfig.from_env({}, root=tmp_path)
    assert c.data_dir == tmp_path / "data" and c.db_path == tmp_path / "data/processed/library.sqlite"
    assert c.files_dir == tmp_path / "data/library" and c.embedding_dir == tmp_path / "artifacts/embeddings"
    assert (c.host, c.port, c.workspace, c.device) == ("127.0.0.1", 8501, "local", "auto")
    assert c.max_upload_bytes == 50 * 1024 * 1024


def test_environment_overrides_and_relative_paths(tmp_path):
    c = AppConfig.from_env({"RLX_DATA_DIR": "mydata", "RLX_DB": "/abs/x.sqlite", "RLX_PORT": "9000", "RLX_MAX_UPLOAD_MB": "5",
                            "RLX_DEVICE": "cpu", "RLX_WORKSPACE": "alice"}, root=tmp_path)
    assert c.data_dir == tmp_path / "mydata" and str(c.db_path) == "/abs/x.sqlite" and c.files_dir == tmp_path / "mydata/library"
    assert (c.port, c.max_upload_mb, c.device, c.workspace) == (9000, 5, "cpu", "alice")


@pytest.mark.parametrize("env", [{"RLX_PORT": "abc"}, {"RLX_PORT": "70000"}, {"RLX_DEVICE": "tpu"}, {"RLX_HOST": "0.0.0.0"},
                                 {"RLX_WORKSPACE": "../evil"}, {"RLX_MAX_UPLOAD_MB": "0"}])
def test_invalid_configuration_fails_loudly(env, tmp_path):
    with pytest.raises(ValueError, match="invalid ResearchLens X configuration"):
        AppConfig.from_env(env, root=tmp_path)
