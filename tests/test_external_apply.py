import pytest

from ronin.cli import apply_ops


class FakeDatabase:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def _configure_wrapper(monkeypatch, workflow):
    database = FakeDatabase()
    monkeypatch.setattr(apply_ops, "load_env", lambda: None)
    monkeypatch.setattr(apply_ops, "load_config", lambda: {"test": True})
    monkeypatch.setattr(apply_ops, "get_db_manager", lambda **_kwargs: database)
    monkeypatch.setattr(apply_ops, "_apply_external_with_db", workflow)
    return database


def test_apply_external_closes_database_after_success(monkeypatch):
    database = _configure_wrapper(monkeypatch, lambda **_kwargs: 0)

    assert apply_ops.apply_external(report=True) == 0
    assert database.close_calls == 1


def test_apply_external_closes_database_after_failure(monkeypatch):
    def fail(**_kwargs):
        raise RuntimeError("workflow failed")

    database = _configure_wrapper(monkeypatch, fail)

    with pytest.raises(RuntimeError, match="workflow failed"):
        apply_ops.apply_external(report=True)
    assert database.close_calls == 1
