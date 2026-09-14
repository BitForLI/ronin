"""Tests for the subscription-backed Codex AI service."""

from types import SimpleNamespace

import openai_codex

from ronin.ai import CodexService


class _FakeThread:
    def __init__(self) -> None:
        self.calls = []

    def run(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(final_response='{"status": "ok"}')


class _FakeCodex:
    instances = []

    def __init__(self, config=None) -> None:
        self.config = config
        self.thread_starts = []
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def account(self):
        account = SimpleNamespace(email="reese@example.com", plan_type="plus")
        return SimpleNamespace(account=account)

    def thread_start(self, **kwargs):
        thread = _FakeThread()
        self.thread_starts.append((kwargs, thread))
        return thread

    def close(self) -> None:
        self.closed = True


def test_codex_service_reuses_thread_and_parses_json(tmp_path, monkeypatch) -> None:
    _FakeCodex.instances.clear()
    monkeypatch.setattr(openai_codex, "Codex", _FakeCodex)
    monkeypatch.setenv("RONIN_CODEX_CWD", str(tmp_path / "codex-workspace"))

    service = CodexService()
    first = service.chat_completion("Return JSON", "First question")
    second = service.chat_completion("Return JSON", "Second question")

    assert first == {"status": "ok"}
    assert second == {"status": "ok"}
    assert len(_FakeCodex.instances) == 1
    assert len(_FakeCodex.instances[0].thread_starts) == 1
    thread = _FakeCodex.instances[0].thread_starts[0][1]
    assert "First question" in thread.calls[0][0]
    assert "Second question" in thread.calls[1][0]
    assert all("<untrusted_external_data>" in call[0] for call in thread.calls)
    assert all(call[1]["effort"] == "low" for call in thread.calls)

    service.close()
    assert _FakeCodex.instances[0].closed is True


def test_codex_service_migrates_old_api_model_names(tmp_path, monkeypatch) -> None:
    _FakeCodex.instances.clear()
    monkeypatch.setattr(openai_codex, "Codex", _FakeCodex)
    monkeypatch.setenv("RONIN_CODEX_CWD", str(tmp_path / "codex-workspace"))

    service = CodexService(default_model="gpt-5.6-luna")
    service.chat_completion(
        "Return JSON",
        "Question",
        model="claude-sonnet-4-6",
    )

    start_options = _FakeCodex.instances[0].thread_starts[0][0]
    assert start_options["model"] == "gpt-5.6-luna"


def test_codex_account_status_uses_chatgpt_login(monkeypatch) -> None:
    _FakeCodex.instances.clear()
    monkeypatch.setattr(openai_codex, "Codex", _FakeCodex)

    ok, message = CodexService.account_status()

    assert ok is True
    assert "ChatGPT" in message
    assert "plus" in message
