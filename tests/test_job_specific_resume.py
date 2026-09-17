"""Tests for project selection and evidence-constrained STAR resume writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from types import SimpleNamespace

from ronin.db import SQLiteManager
from ronin.job_specific_resume import (
    JobSpecificResumeError,
    ProjectFact,
    TailoringResult,
    index_local_projects,
    load_project_catalog,
    load_tailoring_rules,
    build_tailoring_prompts,
    render_latex_projects,
    replace_projects_section,
    select_projects,
    tailor_projects_with_ai,
    validate_tailored_projects,
    write_tailoring_artifacts,
    compile_resume_pdf,
    prepare_precision_resume,
)


def _projects() -> list[ProjectFact]:
    raw = [
        {
            "id": "commerce",
            "name": "Commerce Platform",
            "repository": "https://github.com/example/commerce",
            "summary": "A full-stack ordering platform.",
            "technologies": ["TypeScript", "React", "C#", "PostgreSQL"],
            "capabilities": ["REST API development", "payment workflows"],
            "domains": ["e-commerce"],
            "evidence": [
                {
                    "id": "context:1",
                    "kind": "situation",
                    "text": "Customers needed consistent delivery and pickup ordering workflows.",
                    "source": "README.md",
                },
                {
                    "id": "action:1",
                    "kind": "action",
                    "text": "Built React interfaces and C# API endpoints for product and order workflows.",
                    "source": "frontend/src and backend/src",
                },
                {
                    "id": "result:1",
                    "kind": "result",
                    "text": "The application supports end-to-end product browsing and order creation.",
                    "source": "tests/order-flow.spec.ts",
                },
            ],
            "needs_review": False,
        },
        {
            "id": "telemetry",
            "name": "Telemetry Pipeline",
            "repository": "https://github.com/example/telemetry",
            "summary": "A deterministic streaming analytics project.",
            "technologies": ["Python", "Apache Kafka", "ClickHouse"],
            "capabilities": ["event processing", "data validation"],
            "domains": ["streaming analytics"],
            "evidence": [
                {
                    "id": "context:1",
                    "kind": "task",
                    "text": "Telemetry events needed deterministic validation before aggregation.",
                    "source": "docs/architecture.md",
                },
                {
                    "id": "action:1",
                    "kind": "implementation",
                    "text": "Implemented Python validation and event-time aggregation for telemetry records.",
                    "source": "src/pipeline.py",
                },
                {
                    "id": "result:1",
                    "kind": "outcome",
                    "text": "Repeatable tests separate valid, duplicate, late and malformed records.",
                    "source": "tests/test_pipeline.py",
                },
            ],
            "needs_review": False,
        },
    ]
    return [
        ProjectFact.from_dict(item, index=index) for index, item in enumerate(raw, 1)
    ]


def _valid_payload() -> dict:
    return {
        "projects": [
            {
                "project_id": "commerce",
                "name": "Commerce Platform",
                "technologies": ["React", "TypeScript", "C#"],
                "bullets": [
                    {
                        "text": "Built React interfaces and C# APIs to address inconsistent ordering workflows, enabling end-to-end product browsing and order creation.",
                        "evidence_ids": ["context:1", "action:1", "result:1"],
                    },
                    {
                        "text": "Implemented typed product and order workflows in TypeScript and C#, supporting delivery and pickup ordering through one application.",
                        "evidence_ids": ["context:1", "action:1", "result:1"],
                    },
                ],
            }
        ]
    }


def test_select_projects_prefers_job_relevant_stack() -> None:
    selected = select_projects(
        _projects(),
        job_title="Full-Stack Software Engineering Intern",
        job_description="Build React and TypeScript interfaces and REST APIs for payments.",
        limit=1,
    )
    assert selected[0].project.id == "commerce"
    assert "React" in selected[0].matched_technologies
    assert selected[0].score > 0


def test_unreviewed_projects_are_not_eligible() -> None:
    raw = {
        "id": "draft",
        "name": "Draft",
        "technologies": ["Python"],
        "evidence": [],
        "needs_review": True,
    }
    with pytest.raises(JobSpecificResumeError, match="No reviewed projects"):
        select_projects(
            [ProjectFact.from_dict(raw, 1)],
            "Python Intern",
            "Write Python services",
        )


def test_internship_projects_are_excluded_even_with_review_override() -> None:
    project = ProjectFact.from_dict(
        {
            "id": "internship-alias",
            "name": "Ordering Platform",
            "technologies": ["React"],
            "included_in_experience": True,
        },
        1,
    )
    assert project.included_in_experience
    for allow_unreviewed in (False, True):
        with pytest.raises(JobSpecificResumeError, match="existing work experience"):
            select_projects(
                [project], "React Developer", "React", allow_unreviewed=allow_unreviewed
            )
    matches = select_projects(
        [project, *_projects()], "React Developer", "React TypeScript REST", limit=1
    )
    assert matches[0].project.id == "commerce"


def test_policy_is_reloaded_and_forwarded_to_writer(tmp_path) -> None:
    policy = tmp_path / "rules.md"
    policy.write_text("First policy", encoding="utf-8")
    config = {"precision_apply": {"rules_file": str(policy)}}
    assert load_tailoring_rules(config) == "First policy"
    policy.write_text("Never duplicate internship work", encoding="utf-8")
    rules = load_tailoring_rules(config)
    matches = select_projects(_projects(), "React Intern", "React TypeScript", limit=1)
    captured = []

    class Writer:
        def chat_completion(self, **kwargs):
            captured.append(kwargs["system_prompt"])
            return _valid_payload()

    tailor_projects_with_ai(
        ai_service=Writer(),
        model="test",
        job_id="1",
        job_title="React Intern",
        company="Example",
        job_description="React TypeScript",
        matches=matches,
        bullets_per_project=2,
        tailoring_rules=rules,
    )
    assert rules in captured[0]
    assert "Technical Projects only" in captured[0]
    system, _ = build_tailoring_prompts(
        job_title="React Intern",
        company="Example",
        job_description="React",
        matches=matches,
        tailoring_rules=rules,
    )
    assert rules in system


def test_missing_or_empty_policy_stops_precision_before_ai(tmp_path) -> None:
    policy = tmp_path / "missing.md"
    config = {"precision_apply": {"enabled": True, "rules_file": str(policy)}}
    writer = SimpleNamespace(
        chat_completion=lambda **kw: pytest.fail("Unexpected AI call")
    )
    with pytest.raises(JobSpecificResumeError, match="Cannot read tailoring rules"):
        prepare_precision_resume({}, config, ai_service=writer)
    policy.write_text("  \n", encoding="utf-8")
    with pytest.raises(JobSpecificResumeError, match="file is empty"):
        prepare_precision_resume({}, config, ai_service=writer)


def test_validator_accepts_source_backed_star_copy() -> None:
    matches = select_projects(
        _projects(),
        "Full-Stack Intern",
        "React TypeScript C# REST API",
        limit=1,
    )
    result = validate_tailored_projects(
        _valid_payload(), matches, bullets_per_project=2
    )
    assert result[0].project_id == "commerce"
    assert len(result[0].bullets) == 2


def test_validator_rejects_invented_metric_and_technology() -> None:
    matches = select_projects(
        _projects(),
        "Full-Stack Intern",
        "React TypeScript C# REST API",
        limit=1,
    )
    payload = _valid_payload()
    payload["projects"][0]["bullets"][0]["text"] += " for 10,000 users."
    with pytest.raises(JobSpecificResumeError, match="unsupported numeric claims"):
        validate_tailored_projects(payload, matches, bullets_per_project=2)

    payload = _valid_payload()
    payload["projects"][0]["technologies"].append("Rust")
    with pytest.raises(JobSpecificResumeError, match="unverified technology"):
        validate_tailored_projects(payload, matches, bullets_per_project=2)


def test_ai_writer_retries_after_fact_validation_failure() -> None:
    matches = select_projects(
        _projects(),
        "Full-Stack Intern",
        "React TypeScript C# REST API",
        limit=1,
    )

    class FakeWriter:
        def __init__(self) -> None:
            self.calls = 0

        def chat_completion(self, **_: object) -> dict:
            self.calls += 1
            payload = _valid_payload()
            if self.calls == 1:
                payload["projects"][0]["bullets"][0]["text"] += " for 50,000 users."
            return payload

    writer = FakeWriter()
    result = tailor_projects_with_ai(
        ai_service=writer,
        model="test-model",
        job_id="seek-123",
        job_title="Full-Stack Intern",
        company="Example",
        job_description="React TypeScript C# REST API",
        matches=matches,
        bullets_per_project=2,
    )

    assert writer.calls == 2
    assert result.projects[0].project_id == "commerce"


def test_latex_render_and_section_replacement() -> None:
    matches = select_projects(
        _projects(),
        "Full-Stack Intern",
        "React TypeScript C# REST API",
        limit=1,
    )
    projects = validate_tailored_projects(
        _valid_payload(), matches, bullets_per_project=2
    )
    result = TailoringResult(
        job_id="123",
        job_title="Software Intern",
        company="Example",
        projects=projects,
        matches=tuple(matches),
    )
    rendered = render_latex_projects(result)
    assert r"\section{Technical Projects}" in rendered
    assert r"Technology Stack" not in rendered
    assert r"C\#" in rendered

    base = (
        "\\section{Education}\nEducation\n"
        "\\section{Technical Projects}\nOld project\n"
        "\\section{Skills}\nPython\n"
    )
    replaced = replace_projects_section(base, rendered, ".tex")
    assert "Old project" not in replaced
    assert replaced.count(r"\section{Technical Projects}") == 1
    assert r"\section{Skills}" in replaced


def test_catalog_load_and_artifact_manifest(tmp_path: Path) -> None:
    catalog_path = tmp_path / "projects.yaml"
    catalog_path.write_text(
        """
projects:
  - id: sample
    name: Sample
    repository: https://github.com/example/sample
    technologies: [Python]
    situation: A local tool needed repeatable input handling.
    actions: [Implemented typed Python parsing.]
    results: [The parser handles documented inputs through automated tests.]
    needs_review: false
""".strip(),
        encoding="utf-8",
    )
    projects = load_project_catalog(catalog_path)
    assert [fact.kind for fact in projects[0].evidence] == [
        "situation",
        "action",
        "result",
    ]

    selected = select_projects(
        projects, "Python Intern", "Implement and test Python parsing", limit=1
    )
    payload = {
        "projects": [
            {
                "project_id": "sample",
                "name": "Sample",
                "technologies": ["Python"],
                "bullets": [
                    {
                        "text": "Implemented typed Python parsing for a local tool, producing repeatable input handling covered by automated tests.",
                        "evidence_ids": ["situation:1", "action:1", "result:1"],
                    },
                    {
                        "text": "Built documented input parsing in Python to address inconsistent handling and support repeatable automated verification.",
                        "evidence_ids": ["situation:1", "action:1", "result:1"],
                    },
                ],
            }
        ]
    }
    tailored = validate_tailored_projects(payload, selected, bullets_per_project=2)
    result = TailoringResult(
        job_id="seek-123",
        job_title="Python Intern",
        company="Example",
        projects=tailored,
        matches=tuple(selected),
    )
    artifacts = write_tailoring_artifacts(result=result, output_dir=tmp_path / "out")
    manifest = json.loads(Path(artifacts["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["projects"][0]["id"] == "sample"
    assert Path(artifacts["resume_path"]).exists()


def test_application_ledger_keeps_tailoring_artifacts(tmp_path: Path) -> None:
    database = SQLiteManager(db_path=str(tmp_path / "ronin.db"))
    try:
        assert database.record_application_submission(
            {
                "job_id": "seek-987",
                "title": "Software Engineering Intern",
                "company_name": "Example",
                "source": "seek",
                "url": "https://www.seek.com.au/job/987",
                "selected_projects": json.dumps(["commerce", "telemetry"]),
                "tailored_resume_path": "resumes/example-intern.tex",
                "tailoring_manifest_path": "resumes/example-intern.manifest.json",
                "tailoring_generated_at": "2026-09-13T10:00:00+00:00",
            }
        )
        application = database.get_applications(limit=1)[0]
        assert json.loads(application["selected_projects"]) == [
            "commerce",
            "telemetry",
        ]
        assert application["tailored_resume_path"].endswith("example-intern.tex")
        assert application["tailoring_manifest_path"].endswith(".manifest.json")
    finally:
        database.close()


def test_local_repository_index_is_review_required(tmp_path: Path) -> None:
    repository = tmp_path / "sample-repo"
    repository.mkdir()
    (repository / "README.md").write_text(
        """
# Sample Repository

A small FastAPI service with repeatable request validation.

## Implementation

- Implemented typed endpoints and automated validation
  for incoming records.
""".strip(),
        encoding="utf-8",
    )
    (repository / "app.py").write_text(
        "from fastapi import FastAPI\n", encoding="utf-8"
    )
    (repository / "requirements.txt").write_text("fastapi\npytest\n", encoding="utf-8")

    projects = index_local_projects(tmp_path)

    assert len(projects) == 1
    assert projects[0]["id"] == "sample-repo"
    assert projects[0]["needs_review"] is True
    assert "Python" in projects[0]["technologies"]
    assert "FastAPI" in projects[0]["technologies"]
    action_facts = [
        fact for fact in projects[0]["evidence"] if fact["kind"] == "action"
    ]
    assert action_facts
    assert action_facts[0]["text"].endswith("for incoming records.")


def test_last_latex_section_preserves_document_end() -> None:
    source = r"\begin{document}\section{Technical Projects}old\end{document}"
    assert replace_projects_section(source, "NEW", ".tex").endswith(r"\end{document}")


def test_precision_disabled_spends_no_ai_allowance() -> None:
    assert prepare_precision_resume({}, {"precision_apply": {"enabled": False}}) == {}


def test_precision_missing_template_fails_before_ai() -> None:
    with pytest.raises(JobSpecificResumeError, match="requires catalog"):
        prepare_precision_resume({}, {"precision_apply": {"enabled": True}})


@pytest.mark.parametrize(
    "pages,text,error", [(2, "Resume", "exceeds"), (1, "", "no extractable")]
)
def test_pdf_validation_rejects_bad_output(
    tmp_path, monkeypatch, pages, text, error
) -> None:
    source = tmp_path / "cv.tex"
    source.write_text("LATEX")

    def run(argv, **kwargs):
        assert kwargs["check"] is False
        (tmp_path / "build" / "cv.pdf").write_bytes(b"%PDF-test")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("ronin.job_specific_resume.subprocess.run", run)
    monkeypatch.setattr(
        "pypdf.PdfReader",
        lambda path: SimpleNamespace(
            pages=[SimpleNamespace(extract_text=lambda: text)] * pages
        ),
    )
    with pytest.raises(JobSpecificResumeError, match=error):
        compile_resume_pdf(source, command=["compiler", "{tex}", "{output_dir}"])


def test_compile_failure_does_not_accept_a_pdf(tmp_path, monkeypatch) -> None:
    source = tmp_path / "cv.tex"
    source.write_text("LATEX")
    monkeypatch.setattr(
        "ronin.job_specific_resume.subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stderr="invalid latex", stdout=""
        ),
    )
    with pytest.raises(JobSpecificResumeError, match="invalid latex"):
        compile_resume_pdf(source, command=["compiler"])


def test_prepare_saves_pdf_and_selected_project_snapshot(tmp_path, monkeypatch) -> None:
    import yaml
    import ronin.job_specific_resume as module

    catalog = tmp_path / "catalog.yaml"
    projects = _projects()
    monkeypatch.setattr(module, "load_project_catalog", lambda path: projects)
    catalog.write_text(yaml.safe_dump({"schema_version": 1, "projects": []}))
    base = tmp_path / "base.tex"
    base.write_text(
        r"DUPOON PTY LTD\section{Technical Projects}old\section{Skills}skills\end{document}"
    )
    policy = tmp_path / "policy.md"
    policy.write_text("Never duplicate internship work", encoding="utf-8")

    class Writer:
        def chat_completion(self, **kwargs):
            assert "Never duplicate internship work" in kwargs["system_prompt"]
            return _valid_payload()

    def compile(source, **kwargs):
        assert "DUPOON PTY LTD" in source.read_text()
        assert source.read_text().endswith(r"\end{document}")
        pdf = source.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-test")
        return {
            "resume_pdf_path": str(pdf),
            "resume_text": "DUPOON PTY LTD Commerce Platform",
            "pdf_pages": 1,
        }

    monkeypatch.setattr(module, "compile_resume_pdf", compile)
    config = {
        "precision_apply": {
            "enabled": True,
            "catalog": str(catalog),
            "rules_file": str(policy),
            "base_resume": str(base),
            "output_dir": str(tmp_path / "out"),
            "project_limit": 1,
            "bullets_per_project": 2,
        }
    }
    job = {
        "job_id": "123",
        "title": "Full Stack Intern",
        "company_name": "Example",
        "description": "React TypeScript C# REST APIs",
    }
    result = prepare_precision_resume(job, config, ai_service=Writer())
    assert result["tailored_resume_path"].endswith(".pdf")
    assert json.loads(result["selected_projects"]) == ["commerce"]
    manifest = json.loads(
        Path(result["tailoring_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["pdf_path"] == result["tailored_resume_path"]
    assert manifest["resume_path"].endswith(".tex")
    assert manifest["tailoring_rules"] == "Never duplicate internship work"


def test_metadata_failure_stops_before_application(monkeypatch) -> None:
    from ronin.cli.tailor import prepare_application_resume

    prepared = {
        "selected_projects": '["commerce"]',
        "tailored_resume_path": "new.pdf",
        "tailoring_manifest_path": "new.json",
        "tailoring_generated_at": "now",
        "resume_pdf_path": "new.pdf",
        "resume_text": "CV",
    }
    monkeypatch.setattr(
        "ronin.cli.tailor.prepare_precision_resume", lambda *a, **kw: prepared
    )
    record = {"id": 1}
    with pytest.raises(JobSpecificResumeError, match="could not be saved"):
        prepare_application_resume(
            record, SimpleNamespace(update_record=lambda *a: False), {}
        )
    assert "tailored_resume_path" not in record


def test_batch_skips_failed_tailoring_and_records_actual_pdf(monkeypatch) -> None:
    from ronin.cli import apply_ops, tailor

    calls, submissions = [], []

    class Applier:
        ai_service = None

        def login(self):
            return True

        def cleanup(self):
            pass

        def apply_to_job(self, **kwargs):
            calls.append(kwargs)
            return "APPLIED"

    def prepare(record, db, config, **kwargs):
        if record["id"] == 1:
            raise JobSpecificResumeError("Compilation failed")
        record.update(selected_projects='["commerce"]', tailored_resume_path="new.pdf")
        return {"resume_pdf_path": "new.pdf", "resume_text": "Commerce Platform"}

    monkeypatch.setattr(apply_ops, "SeekApplier", Applier)
    monkeypatch.setattr(apply_ops, "load_config", lambda: {})
    monkeypatch.setattr(tailor, "prepare_application_resume", prepare)
    db = SimpleNamespace(
        update_record=lambda *a: True,
        mark_job_applied=lambda **kw: True,
        record_application_submission=lambda payload: submissions.append(payload),
    )
    result = apply_ops._apply_records(
        [{"id": 1, "title": "Bad"}, {"id": 2, "title": "Good"}],
        db,
        "builder",
        None,
        "default",
        "old-hash",
    )
    assert result == {"applied": 1, "failed": 1, "stale": 0}
    assert len(calls) == 1 and calls[0]["resume_pdf_path"] == "new.pdf"
    assert submissions[0]["tailored_resume_path"] == "new.pdf"
    assert submissions[0]["resume_variant_sent"] == "job-specific"
    assert submissions[0]["resume_commit_hash"] is None


def test_cover_letters_use_uploaded_resume_and_each_company(monkeypatch) -> None:
    from ronin.applier import cover_letter as module
    from ronin.profile import Profile

    calls = []
    generator = module.CoverLetterGenerator.__new__(module.CoverLetterGenerator)
    generator.profile = Profile()
    generator.model = "test"
    generator.ai_service = SimpleNamespace(
        chat_completion=lambda **kwargs: calls.append(kwargs)
        or {"response": kwargs["user_message"]}
    )
    monkeypatch.setattr(
        module, "generate_cover_letter_prompt", lambda **kwargs: kwargs["resume_text"]
    )
    monkeypatch.setattr(
        Profile,
        "get_highlights_text",
        lambda self: pytest.fail(
            "Must not replace explicit tailored resume with old highlights"
        ),
    )
    first = generator.generate_cover_letter(
        "React JD", "Graduate", "Alpha", "job-specific", resume_text="Uploaded Alpha CV"
    )
    second = generator.generate_cover_letter(
        "Python JD", "Intern", "Beta", "job-specific", resume_text="Uploaded Beta CV"
    )
    assert calls[0]["system_prompt"] == "Uploaded Alpha CV"
    assert calls[1]["system_prompt"] == "Uploaded Beta CV"
    assert "Alpha" in first["response"] and "Beta" in second["response"]
    assert first != second
