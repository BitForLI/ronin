from pathlib import Path
from types import SimpleNamespace

from ronin.applier.agent_applier import AgentApplier, _safe_company_filename


def _applier():
    applier = AgentApplier.__new__(AgentApplier)
    applier.profile = SimpleNamespace(
        personal=SimpleNamespace(
            name="Reese Lee",
            email="reese@example.com",
            phone="0400000000",
            location="Sydney, NSW",
        ),
        work_rights=SimpleNamespace(
            citizenship="Chinese citizen",
            visa_status="Australian Student visa (subclass 500)",
            has_drivers_license=False,
            willing_to_relocate=False,
            notice_period="Immediate",
        ),
        professional=SimpleNamespace(years_experience=0),
    )
    applier._resume_text = "GitHub: https://github.com/example/repo"
    return applier


def _resume_profile(filename: str):
    resume = SimpleNamespace(file=filename)
    return SimpleNamespace(
        resumes=[resume],
        get_resume=lambda _name: resume,
    )


def test_known_answers_use_saved_profile_facts():
    applier = _applier()

    assert applier._known_answer("First name", "text") == "Reese"
    assert applier._known_answer("Family name", "text") == "Lee"
    assert applier._known_answer("Email address", "email") == "reese@example.com"
    assert applier._known_answer("Current visa", "text") == (
        "Australian Student visa (subclass 500)"
    )
    assert applier._known_answer("Years of experience", "text") == "0"


def test_work_rights_choices_do_not_claim_unrestricted_status():
    applier = _applier()

    assert (
        applier._choice_preference("Do you have unrestricted work rights?")[0] == "no"
    )
    assert applier._choice_preference("What is your current visa?")[0] == (
        "student visa"
    )
    assert applier._choice_preference(
        "Will you now or in the future require sponsorship?"
    ) == ["yes"]
    assert applier._choice_preference("Do you have full work rights?")[0] == "no"
    assert applier._choice_preference("Do you have full working rights?")[0] == "no"


def test_resume_lookup_uses_configured_ronin_home(tmp_path, monkeypatch):
    ronin_home = tmp_path / "ronin-home"
    resumes = ronin_home / "resumes"
    resumes.mkdir(parents=True)
    resume_text = resumes / "general.txt"
    resume_pdf = resumes / "general.pdf"
    resume_text.write_text("General resume", encoding="utf-8")
    resume_pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("RONIN_HOME", str(ronin_home))

    applier = AgentApplier.__new__(AgentApplier)
    applier.profile = _resume_profile("general.txt")
    applier.agent_config = {"resume_pdf_dir": str(tmp_path / "missing")}

    assert applier._resolve_resume_text("default") == "General resume"
    assert applier._resolve_resume_pdf("default") == Path(resume_pdf).resolve()


def test_company_archive_filename_is_windows_safe():
    assert _safe_company_filename("ACME: Cloud / AI? Pty Ltd") == (
        "ACME Cloud AI Pty Ltd"
    )
