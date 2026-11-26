"""How specdec/common.py finds Hugging Face checkpoints (CPU, no network).

Regression: `bench --target mla-gpt2-medium --draft distilgpt2` failed off Bridges-2 because
common.py forced HF_HOME to the Bridges cache and only looked at local files.
"""
import os

import pytest

hub = pytest.importorskip("huggingface_hub")

from specdec import common  # noqa: E402

try:
    from huggingface_hub.errors import LocalEntryNotFoundError
except ImportError:  # huggingface_hub < 0.23
    from huggingface_hub.utils import LocalEntryNotFoundError


@pytest.mark.parametrize("preset, bridges_exists, expected", [
    (None, False, None),            # laptop/desktop: keep HF's default cache
    (None, True, "bridges"),        # Bridges-2: shared cache
    ("/my/cache", True, "/my/cache"),  # explicit HF_HOME always wins
])
def test_hf_home_defaults_to_bridges_cache_only_on_bridges(monkeypatch, tmp_path, preset, bridges_exists, expected):
    bridges = tmp_path / "bridges_hf"
    if bridges_exists:
        bridges.mkdir()
    monkeypatch.setattr(common, "BRIDGES_HF_HOME", str(bridges))
    if preset is None:
        monkeypatch.delenv("HF_HOME", raising=False)
    else:
        monkeypatch.setenv("HF_HOME", preset)
    common._use_bridges_hf_cache()
    assert os.environ.get("HF_HOME") == (str(bridges) if expected == "bridges" else expected)


def _fake_hub(monkeypatch, cached_files):
    """Patch snapshot_download: the local cache holds `cached_files` (None = repo not cached)."""
    calls = []

    def snapshot_download(repo_id, local_files_only=False, allow_patterns=None, **kw):
        calls.append({"local_files_only": local_files_only, "allow_patterns": allow_patterns})
        if local_files_only:
            if cached_files is None:
                raise LocalEntryNotFoundError("not cached")
            return str(cached_files)
        return "/downloaded/" + repo_id

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    return calls


def test_model_dir_uses_complete_local_snapshot(monkeypatch, tmp_path):
    for f in common.HF_FILES:
        (tmp_path / f).write_text("{}")
    calls = _fake_hub(monkeypatch, tmp_path)
    assert common.model_dir("distilgpt2") == str(tmp_path)
    assert calls == [{"local_files_only": True, "allow_patterns": None}]


@pytest.mark.parametrize("cached", ["missing", "partial"])
def test_model_dir_downloads_only_needed_files_when_not_cached(monkeypatch, tmp_path, cached):
    if cached == "partial":
        (tmp_path / "config.json").write_text("{}")  # e.g. cached by another tool without weights
    calls = _fake_hub(monkeypatch, None if cached == "missing" else tmp_path)
    assert common.model_dir("distilgpt2") == "/downloaded/" + common.MODELS["distilgpt2"]
    assert calls[-1] == {"local_files_only": False, "allow_patterns": list(common.HF_FILES)}
