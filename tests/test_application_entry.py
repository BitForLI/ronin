"""SEEK-native forms and external Apply with SEEK consent are different flows."""

from types import SimpleNamespace

import pytest

import ronin.applier.applier as module
from ronin.applier.applier import SeekApplier


class EntryButton:
    def __init__(self, driver, visible=True):
        self.driver = driver
        self.visible = visible
        self.clicks = 0

    def is_displayed(self):
        return self.visible

    def is_enabled(self):
        return True

    def click(self):
        self.clicks += 1
        self.driver.window_handles.append("consent")


class EntryDriver:
    def __init__(self, url, *, native=False, external_form=False, buttons=0):
        self.current_url = url
        self.native = native
        self.external_form = external_form
        self.window_handles = ["original"]
        self.waits = []
        self.buttons = [EntryButton(self) for _ in range(buttons)]
        self.switch_to = SimpleNamespace(window=self.switch_window)

    def switch_window(self, handle):
        assert handle == "consent"
        self.current_url = "https://au.seek.com/awsk/authorize?private=value"

    def implicitly_wait(self, duration):
        self.waits.append(duration)

    def find_elements(self, by, selector):
        if selector == SeekApplier.RESUME_RADIO_SELECTOR:
            return [object()] if self.native else []
        if selector == SeekApplier.SEEK_PROFILE_ENTRY_SELECTOR:
            return self.buttons
        if selector == "form input, form select":
            return [object()] if self.external_form else []
        return []


def applier_for(driver):
    applier = SeekApplier.__new__(SeekApplier)
    applier.chrome_driver = SimpleNamespace(
        driver=driver,
        current_url=driver.current_url,
        is_logged_in=True,
        initialize=lambda: None,
    )
    applier.application_entry_reason = ""
    return applier


@pytest.mark.parametrize(
    "url,options,expected",
    [
        ("https://au.seek.com/job/1/apply", {"native": True}, "NATIVE"),
        ("https://au.seek.com/job/1/apply", {}, ""),
        ("https://au.seek.com/job/1/apply/external", {}, "EXTERNAL_VISIT"),
        ("https://au.seek.com/awsk/authorize", {}, "SEEK_AUTHORIZATION"),
        ("https://login.seek.com/login", {}, "LOGIN_REQUIRED"),
        ("https://accounts.google.com/signin", {}, "LOGIN_REQUIRED"),
        ("https://applr.io/l/example", {"buttons": 1}, "SEEK_PROFILE_ENTRY"),
        ("https://applr.io/l/example", {"external_form": True}, "EXTERNAL_FORM"),
        # An employer's resume radios do not make it a SEEK-native form.
        ("https://employer.example/form", {"native": True}, ""),
    ],
)
def test_entry_state(url, options, expected):
    driver = EntryDriver(url, **options)
    assert applier_for(driver)._application_entry_state(driver) == expected


class ImmediateWait:
    def __init__(self, driver, timeout):
        self.driver = driver

    def until(self, condition):
        result = condition(self.driver)
        assert result, "Expected a ready entry"
        return result


def test_apply_with_seek_opens_consent_but_never_grants_access(monkeypatch):
    monkeypatch.setattr(module, "WebDriverWait", ImmediateWait)
    driver = EntryDriver("https://applr.io/l/example", buttons=1)
    applier = applier_for(driver)
    assert applier._resolve_application_entry() == "NEEDS_HUMAN"
    assert driver.buttons[0].clicks == 1
    assert "DEFAULT resume" in applier.application_entry_reason
    assert "NOT been submitted" in applier.application_entry_reason
    assert driver.waits == [0, 10]


def test_multiple_entries_are_not_guessed(monkeypatch):
    monkeypatch.setattr(module, "WebDriverWait", ImmediateWait)
    driver = EntryDriver("https://applr.io/l/example", buttons=2)
    assert applier_for(driver)._resolve_application_entry() == "NEEDS_HUMAN"
    assert not any(button.clicks for button in driver.buttons)


def test_hidden_entry_does_not_override_external_form():
    driver = EntryDriver("https://applr.io/l/example", buttons=1, external_form=True)
    driver.buttons[0].visible = False
    assert applier_for(driver)._application_entry_state(driver) == "EXTERNAL_FORM"


def test_native_form_remains_on_existing_resume_flow(monkeypatch):
    monkeypatch.setattr(module, "WebDriverWait", ImmediateWait)
    driver = EntryDriver("https://au.seek.com/job/1/apply", native=True)
    assert applier_for(driver)._resolve_application_entry() is None


@pytest.mark.parametrize("path", ["/job/1/apply/external", "/awsk/authorize"])
def test_external_visit_and_consent_are_never_success(path):
    driver = EntryDriver("https://au.seek.com" + path)
    applier = applier_for(driver)
    # No source is needed: these routes must be rejected before success scanning.
    assert applier._check_success() is False


def test_needs_human_stops_before_resume_upload(monkeypatch):
    applier = applier_for(EntryDriver("https://applr.io/l/example"))
    applier.config = {"precision_apply": {"enabled": False}}
    applier.question_handler = SimpleNamespace(
        ai_handler=SimpleNamespace(resume_text_override=None)
    )
    monkeypatch.setattr(applier, "_navigate_to_job", lambda job_id: "NEEDS_HUMAN")
    monkeypatch.setattr(
        applier, "_handle_resume", lambda **kwargs: pytest.fail("Must not upload")
    )
    assert (
        applier.apply_to_job("1", "JD", 0, "Python", "Company", "Junior")
        == "NEEDS_HUMAN"
    )
