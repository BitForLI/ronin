"""Tests for project selection and evidence-constrained STAR resume writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ronin.db import SQLiteManager
from ronin.job_specific_resume import (
    JobSpecificResumeError,
    ProjectFact,
    TailoringResult,
    index_local_projects,
    load_project_catalog,
    render_latex_projects,
    replace_projects_section,
    select_projects,
    tailor_projects_with_ai,
    validate_tailored_projects,
    write_tailoring_artifacts,
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
