"""Full-resume targeting, source freshness and actual-layout policy tests."""

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from ronin.job_specific_resume import (
    JobSpecificResumeError,
    ProjectFact,
    TailoringResult,
    score_project,
    validate_tailored_projects,
)
from ronin.precision_resume import (
    balance_resume_margins,
    inspect_resume_layout,
    job_prompt_context,
    read_base_skills,
    read_project_sources,
    repair_resume_layout,
    semantic_project_selection,
    tailor_resume_sections,
    verify_resume_claims,
)

WORKSPACE = Path(__file__).resolve().parents[2]


def project(pid="p", experience=False):
    return ProjectFact.from_dict(
        {
            "id": pid,
            "name": pid,
            "technologies": ["React", "Python"],
            "included_in_experience": experience,
            "local_path": pid,
            "reviewed_commit": "abc",
            "evidence": [
                {
                    "id": "context",
                    "kind": "situation",
                    "text": "Customers needed consistent ordering.",
                },
                {
                    "id": "action",
                    "kind": "implementation",
                    "text": "Built React interfaces for ordering.",
                },
                {
                    "id": "result",
                    "kind": "result",
                    "text": "Enabled consistent ordering.",
                },
            ],
        },
        1,
    )


JOB = {
    "title": "Graduate Developer",
    "company_name": "Example",
    "description": "Build React interfaces and Python APIs.",
}


def test_job_prompts_omit_binary_database_vectors():
    record = dict(JOB, embedding=b"\x00\x01", private_metadata={"ignored": True})
    context = job_prompt_context(record)
    assert context["description"] == JOB["description"]
    assert "embedding" not in context and "private_metadata" not in context
    assert json.loads(json.dumps(context)) == context


@pytest.mark.parametrize("amount,expected", [(0, "Negotiable"), (80000, "AUD $80000")])
def test_salary_zero_means_unspecified_not_free_work(amount, expected):
    from ronin.applier.ai_handler import AIResponseHandler

    handler = AIResponseHandler.__new__(AIResponseHandler)
    handler.profile = SimpleNamespace(
        professional=SimpleNamespace(salary_min=amount, salary_max=amount)
    )
    handler.config = {}
    response = handler._process_ai_response(
        {"response": f"AUD ${amount}"},
        {"type": "textarea", "question": "What is your expected annual salary?"},
    )
    assert response == {"response": expected}
    assert handler._process_ai_response(
        {"response": "AUD $0"},
        {"type": "textarea", "question": "What is your current salary?"},
    ) == {"response": "AUD $0"}


def selection(pid="p"):
    return {
        "requirements": [
            {
                "id": "r1",
                "keyword": "React",
                "priority": "required",
                "quote": "React interfaces",
            }
        ],
        "selected_projects": [
            {"id": pid, "reason": "Verified ordering interface contribution"}
        ],
        "coverage_gaps": [],
    }


def writer(payload):
    return SimpleNamespace(chat_completion=lambda **kw: payload)


def test_semantic_selection_and_exact_jd_quotes():
    matches, analysis = semantic_project_selection(
        [project(), project("intern", True)],
        JOB,
        ai_service=writer(selection()),
        model="test",
        limit=1,
    )
    assert matches[0].project.id == "p"
    assert analysis["requirements"][0]["quote"] in JOB["description"]


@pytest.mark.parametrize("pid", ["unknown", "intern"])
def test_semantic_selection_rejects_unknown_or_experience_projects(pid):
    with pytest.raises(JobSpecificResumeError, match="unknown"):
        semantic_project_selection(
            [project(), project("intern", True)],
            JOB,
            ai_service=writer(selection(pid)),
            model="test",
            limit=1,
        )


def test_semantic_selection_rejects_invented_requirement():
    payload = selection()
    payload["requirements"][0]["quote"] = "Rust microservices"
    with pytest.raises(JobSpecificResumeError, match="exact JD"):
        semantic_project_selection(
            [project()], JOB, ai_service=writer(payload), model="test", limit=1
        )


def test_duplicate_selection_is_rejected():
    payload = selection()
    payload["selected_projects"] *= 2
    with pytest.raises(JobSpecificResumeError, match="duplicate"):
        semantic_project_selection(
            [project(), project("q")],
            JOB,
            ai_service=writer(payload),
            model="test",
            limit=2,
        )


def test_coverage_gap_references_a_real_requirement():
    payload = selection()
    payload["coverage_gaps"] = [{"requirement_id": "missing", "reason": "gap"}]
    with pytest.raises(JobSpecificResumeError, match="coverage"):
        semantic_project_selection(
            [project()], JOB, ai_service=writer(payload), model="test", limit=1
        )


def test_sources_use_reviewed_git_blobs_and_not_working_tree(tmp_path, monkeypatch):
    repo = tmp_path / "p"
    repo.mkdir()
    (repo / "README.md").write_text("unreviewed prose")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "rev-parse" in argv:
            return SimpleNamespace(returncode=0, stdout="abc\n", stderr="")
        return SimpleNamespace(returncode=0, stdout=b"Reviewed project documentation")

    monkeypatch.setattr("ronin.precision_resume.subprocess.run", run)
    docs = read_project_sources([project()], tmp_path / "catalog.yaml")
    assert docs["p"]["files"][0]["text"] == "Reviewed project documentation"
    assert any("abc:README.md" in c for c in calls)


def test_code_change_stops_source_reading(tmp_path, monkeypatch):
    (tmp_path / "p").mkdir()

    def run(argv, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="new" if "rev-parse" in argv else "src/api.py\n",
            stderr="",
        )

    monkeypatch.setattr("ronin.precision_resume.subprocess.run", run)
    with pytest.raises(JobSpecificResumeError, match="code changed"):
        read_project_sources([project()], tmp_path / "catalog.yaml")


def test_git_error_is_not_silently_treated_as_a_version(tmp_path, monkeypatch):
    (tmp_path / "p").mkdir()
    monkeypatch.setattr(
        "ronin.precision_resume.subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            returncode=1, stdout="", stderr="ownership error"
        ),
    )
    with pytest.raises(JobSpecificResumeError, match="ownership error"):
        read_project_sources([project()], tmp_path / "catalog.yaml")


def section_payload(p, skills):
    return {
        "experience": {
            "project_id": p.id,
            "name": p.name,
            "technologies": ["React", "Python"],
            "bullets": [
                {
                    "text": "Built React ordering interfaces to address inconsistent "
                    "customer workflows, enabling consistent ordering.",
                    "evidence_ids": ["context", "action", "result"],
                }
            ]
            * 3,
        },
        "skills": skills,
    }


def test_sections_change_only_bullets_stack_and_skills():
    base = (WORKSPACE / "main_comprehensive_software_engineer_resume.tex").read_text(
        encoding="utf-8"
    )
    p = project("intern", True)
    skills = read_base_skills(base)
    assert "C#" in skills["Languages"]
    skills["Languages"] = ["TypeScript", "Python"]
    changed, snapshot = tailor_resume_sections(
        base,
        p,
        [score_project(project(), JOB["description"])],
        JOB,
        {},
        {},
        ai_service=writer(section_payload(p, skills)),
        model="test",
        rules="No invented claims",
    )
    for literal in [
        "DUPOON PTY LTD",
        "Jan 2026 -- Jul 2026",
        "Student visa; 485 eligible",
        "Master of Computer Science",
    ]:
        assert literal in changed
    assert (
        base.split(r"\section{Intern Experience}")[0]
        == changed.split(r"\section{Intern Experience}")[0]
    )
    assert "Languages: & TypeScript, Python" in changed
    assert snapshot["experience"]["project_id"] == "intern"
    assert (
        base.split(r"\section{Technical Projects}")[1].split(
            r"\section{Technical Skills}"
        )[0]
        == changed.split(r"\section{Technical Projects}")[1].split(
            r"\section{Technical Skills}"
        )[0]
    )


def test_sections_reject_unverified_skills():
    base = (WORKSPACE / "main_comprehensive_software_engineer_resume.tex").read_text(
        encoding="utf-8"
    )
    p = project("intern", True)
    skills = read_base_skills(base)
    skills["Languages"] = ["Rust"]
    with pytest.raises(JobSpecificResumeError, match="Unverified"):
        tailor_resume_sections(
            base,
            p,
            [],
            JOB,
            {},
            {},
            ai_service=writer(section_payload(p, skills)),
            model="test",
            rules="",
        )


def test_skills_can_include_verified_selected_project_technology():
    base = (WORKSPACE / "main_comprehensive_software_engineer_resume.tex").read_text(
        encoding="utf-8"
    )
    p = project("intern", True)
    selected = replace(project(), technologies=("Databricks",))
    skills = read_base_skills(base)
    skills["Data and Messaging"] = ["Databricks"]
    changed, _ = tailor_resume_sections(
        base,
        p,
        [score_project(selected, JOB["description"])],
        JOB,
        {},
        {},
        ai_service=writer(section_payload(p, skills)),
        model="test",
        rules="",
    )
    assert "Data and Messaging: & Databricks" in changed


def test_standalone_build_uses_full_preparation_without_loading_profile(
    tmp_path, monkeypatch
):
    import ronin.ai as ai
    from ronin.cli import tailor

    captured = []
    monkeypatch.setattr(tailor, "load_env", lambda: None)
    monkeypatch.setattr(
        tailor, "_load_job", lambda **kw: (dict(JOB, job_id="qa"), None, None)
    )
    monkeypatch.setattr(
        tailor, "load_config", lambda: {"precision_apply": {"full_resume": True}}
    )
    monkeypatch.setattr(
        tailor,
        "load_profile",
        lambda: pytest.fail("Must not reload/recreate candidate profile"),
    )
    monkeypatch.setattr(
        ai, "CodexService", lambda **kw: SimpleNamespace(close=lambda: None)
    )

    def prepare(job, config, **kw):
        captured.append(config["precision_apply"])
        return {"resume_pdf_path": "qa.pdf", "tailoring_manifest_path": "qa.json"}

    monkeypatch.setattr(tailor, "prepare_precision_resume", prepare)
    assert tailor.build_resume(job_id="qa", output_dir=str(tmp_path)) == 0
    assert captured[0]["enabled"] is True
    assert captured[0]["output_dir"] == str(tmp_path.resolve())


def fake_page(monkeypatch, line_values, top=20, bottom=20):
    words = []
    for i, (text, last_x) in enumerate(line_values):
        tokens = text.split()
        for j, token in enumerate(tokens):
            words.append(
                {
                    "text": token,
                    "top": top + 15 * i,
                    "x0": 30 + j * 40,
                    "x1": last_x if j == len(tokens) - 1 else 60 + j * 40,
                }
            )
    page = SimpleNamespace(
        width=600,
        height=800,
        extract_words=lambda **kw: words,
        chars=[{"text": "x", "top": top, "bottom": 800 - bottom, "x0": 30, "x1": 550}],
    )

    class Doc:
        pages = [page]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("pdfplumber.open", lambda *a: Doc())


def test_layout_rejects_short_third_line(monkeypatch):
    fake_page(
        monkeypatch,
        [
            ("Built React interfaces", 550),
            ("for customer ordering", 550),
            ("and checkout", 100),
        ],
    )
    layout = inspect_resume_layout(
        Path("unused.pdf"),
        ["Built React interfaces for customer ordering and checkout"],
    )
    assert any("third line" in issue for issue in layout["issues"])


def test_layout_accepts_two_lines_and_filled_third_line(monkeypatch):
    fake_page(
        monkeypatch, [("Built React interfaces", 550), ("for customer ordering", 480)]
    )
    assert not inspect_resume_layout(
        Path("unused.pdf"), ["Built React interfaces for customer ordering"]
    )["issues"]
    fake_page(
        monkeypatch,
        [
            ("Built React interfaces", 550),
            ("for customer ordering", 550),
            ("through typed checkout workflows", 520),
        ],
    )
    assert not inspect_resume_layout(
        Path("unused.pdf"),
        [
            "Built React interfaces for customer ordering "
            "through typed checkout workflows"
        ],
    )["issues"]


def test_layout_rejects_missing_bullet_and_imbalanced_margins(monkeypatch):
    fake_page(monkeypatch, [("Built ordering interfaces", 550)], top=20, bottom=120)
    issues = inspect_resume_layout(Path("unused.pdf"), ["Different sentence"])["issues"]
    assert any("cannot locate" in i for i in issues)
    assert any("whitespace" in i for i in issues)


def test_internship_cannot_be_shortened_to_two_lines(monkeypatch):
    fake_page(
        monkeypatch, [("Built React interfaces", 550), ("for customer ordering", 500)]
    )
    result = inspect_resume_layout(
        Path("unused.pdf"),
        ["Built React interfaces for customer ordering"],
        experience_count=1,
    )
    assert any("needs three filled lines" in issue for issue in result["issues"])


def test_skills_must_fit_one_category_per_line(monkeypatch):
    fake_page(monkeypatch, [("Languages Python", 550), ("SQL TypeScript", 500)])
    result = inspect_resume_layout(
        Path("unused.pdf"), [], skills={"Languages": ["Python", "SQL", "TypeScript"]}
    )
    assert any("Skills category" in issue for issue in result["issues"])


def test_margin_balance_keeps_font_and_horizontal_margins():
    template = (
        r"\geometry{left=0.65cm,top=0.5cm,right=0.65cm,bottom=0.5cm}"
        r"\fontsize{12.15pt}{15.1pt}"
    )
    balanced = balance_resume_margins(template, {"top_margin": 20, "bottom_margin": 80})
    assert "top=44.17pt" in balanced
    assert "left=0.65cm" in balanced and "right=0.65cm" in balanced
    assert r"\fontsize{12.15pt}{15.1pt}" in balanced


def test_semantic_claim_check_rejects_overstatement():
    result = SimpleNamespace(projects=[])
    with pytest.raises(JobSpecificResumeError, match="rejected"):
        verify_resume_claims(
            result,
            {},
            [],
            {},
            ai_service=writer({"approved": False, "issues": ["Deployment not proved"]}),
            model="test",
        )
    assert verify_resume_claims(
        result,
        {},
        [],
        {},
        ai_service=writer({"approved": True, "issues": []}),
        model="test",
    )["approved"]


def test_targeted_repair_freezes_every_other_bullet():
    base = (WORKSPACE / "main_comprehensive_software_engineer_resume.tex").read_text(
        encoding="utf-8"
    )
    independent, intern = project(), project("intern", True)
    match = score_project(independent, JOB["description"])
    skills = read_base_skills(base)
    draft = section_payload(independent, skills)["experience"]
    generated = validate_tailored_projects({"projects": [draft]}, [match])
    result = TailoringResult("qa", JOB["title"], "Example", generated, (match,))
    snapshot = section_payload(intern, skills)
    replacement = (
        "Built React ordering interfaces to address inconsistent customer "
        "workflows, connecting ordering screens through one consistent "
        "customer workflow."
    )
    captured = []
    ai = SimpleNamespace(
        chat_completion=lambda **kw: captured.append(kw)
        or {
            "replacements": [
                {
                    "index": 3,
                    "text": replacement,
                    "evidence_ids": ["action"],
                }
            ],
        }
    )
    changed, sections, text = repair_resume_layout(
        result,
        snapshot,
        {
            "issues": ["Bullet 3: third line too short"],
            "bullets": [{"index": 3, "lines": 3, "last_line_fill": 0.65}],
        },
        base,
        base,
        intern,
        {},
        ai_service=ai,
        model="test",
        max_words=65,
    )
    assert changed.projects == result.projects
    assert (
        sections["experience"]["bullets"][:2] == snapshot["experience"]["bullets"][:2]
    )
    assert sections["experience"]["bullets"][2]["text"] == replacement
    assert "context" in sections["experience"]["bullets"][2]["evidence_ids"]
    assert sections["skills"] == snapshot["skills"]
    assert "DUPOON PTY LTD" in text and "Student visa" in text
    payload = json.loads(captured[0]["user_message"])
    requests = payload["requests"]
    assert len(requests) == 1 and requests[0]["index"] == 3
    assert len(payload["other_experience_bullets"]) == 2

    repair_resume_layout(
        result,
        snapshot,
        {
            "issues": ["Bullet 3: internship currently has 2 lines"],
            "bullets": [{"index": 3, "lines": 2, "last_line_fill": 0.3}],
        },
        base,
        base,
        intern,
        {},
        ai_service=ai,
        model="test",
        max_words=65,
    )
    short_bullet_request = json.loads(captured[1]["user_message"])["requests"][0]
    assert short_bullet_request["approx_target_characters"] <= round(
        len(snapshot["experience"]["bullets"][2]["text"]) * 1.2
    )


def test_length_retry_preserves_accepted_skills_and_header():
    base = (WORKSPACE / "main_comprehensive_software_engineer_resume.tex").read_text(
        encoding="utf-8"
    )
    p, intern = project(), project("intern", True)
    match = score_project(p, JOB["description"])
    skills = read_base_skills(base)
    snapshot = section_payload(intern, skills)
    generated = validate_tailored_projects(
        {"projects": [section_payload(p, skills)["experience"]]}, [match]
    )
    result = TailoringResult("qa", JOB["title"], "Example", generated, (match,))
    narrowed = {label: values[:1] for label, values in skills.items()}
    replacement = {
        "index": 3,
        "text": snapshot["experience"]["bullets"][2]["text"],
        "evidence_ids": ["action"],
    }
    replies = iter(
        [
            {
                "replacements": [replacement],
                "skills": narrowed,
                "experience_technologies": ["Python", "React"],
            },
            {"replacements": [replacement]},
        ]
    )
    _, sections, _ = repair_resume_layout(
        result,
        snapshot,
        {
            "issues": [
                "Bullet 3: internship currently has 2 lines",
                "Skills category Languages: keep to one rendered line",
                "Text extends outside the page",
            ],
            "bullets": [{"index": 3, "lines": 2, "last_line_fill": 0.3}],
        },
        base,
        base,
        intern,
        {},
        ai_service=SimpleNamespace(chat_completion=lambda **kw: next(replies)),
        model="test",
        max_words=65,
    )
    assert sections["skills"] == narrowed
    assert sections["experience"]["technologies"] == ["Python", "React"]
    assert "context" in sections["experience"]["bullets"][2]["evidence_ids"]


@pytest.mark.parametrize("status,quick_apply", [("APPLIED", 1), ("DISCOVERED", 0)])
def test_apply_one_refuses_applied_and_external_jobs(monkeypatch, status, quick_apply):
    import ronin.cli.apply_ops as module

    record = {"status": status, "quick_apply": quick_apply}
    closed = []
    db = SimpleNamespace(
        get_job_by_job_id=lambda job_id: record,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(module, "load_env", lambda: None)
    monkeypatch.setattr(
        module, "load_config", lambda: {"precision_apply": {"enabled": True}}
    )
    monkeypatch.setattr(module, "get_db_manager", lambda **kw: db)
    monkeypatch.setattr(
        module, "_apply_records", lambda **kw: pytest.fail("must not apply")
    )
    assert module.apply_one("123", yes=True) == 1
    assert closed == [True]


def test_apply_one_passes_exactly_one_job_to_precision_pipeline(monkeypatch):
    import ronin.cli.apply_ops as module

    record = {
        "id": 55,
        "job_id": "123",
        "status": "DISCOVERED",
        "quick_apply": 1,
        "archetype_primary": "builder",
    }
    captured = []
    db = SimpleNamespace(get_job_by_job_id=lambda job_id: record, close=lambda: None)
    monkeypatch.setattr(module, "load_env", lambda: None)
    monkeypatch.setattr(
        module, "load_config", lambda: {"precision_apply": {"enabled": True}}
    )
    monkeypatch.setattr(module, "get_db_manager", lambda **kw: db)
    monkeypatch.setattr(
        module,
        "_apply_records",
        lambda **kw: captured.append(kw) or {"applied": 1, "stale": 0, "failed": 0},
    )
    assert module.apply_one("123", yes=True) == 0
    assert captured[0]["jobs"] == [record]
    assert captured[0]["resume_variant_sent"] == "job-specific"


def test_targeted_repair_rejects_changes_to_another_bullet():
    base = (WORKSPACE / "main_comprehensive_software_engineer_resume.tex").read_text(
        encoding="utf-8"
    )
    p, intern = project(), project("intern", True)
    match = score_project(p, JOB["description"])
    skills = read_base_skills(base)
    generated = validate_tailored_projects(
        {"projects": [section_payload(p, skills)["experience"]]}, [match]
    )
    result = TailoringResult("qa", JOB["title"], "Example", generated, (match,))
    with pytest.raises(JobSpecificResumeError, match="wrong bullet set"):
        repair_resume_layout(
            result,
            section_payload(intern, skills),
            {"issues": ["Bullet 3: too short"]},
            base,
            base,
            intern,
            {},
            ai_service=writer({"replacements": [{"index": 1, "text": "Bad"}]}),
            model="test",
            max_words=65,
        )


def test_full_preparation_retries_layout_without_rewriting_identity(
    tmp_path, monkeypatch
):
    import ronin.job_specific_resume as module
    import ronin.precision_resume as full

    projects = [project(), project("intern", True)]
    monkeypatch.setattr(module, "load_project_catalog", lambda path: projects)
    base = WORKSPACE / "main_comprehensive_software_engineer_resume.tex"
    skills = read_base_skills(base.read_text(encoding="utf-8"))
    draft = section_payload(projects[1], skills)["experience"]
    draft["project_id"] = "p"
    draft["name"] = "p"
    draft["bullets"] = draft["bullets"][:2]

    class AI:
        def chat_completion(self, **kw):
            system = kw["system_prompt"]
            if system.startswith("Analyse"):
                return selection()
            if system.startswith("Rewrite the internship"):
                return section_payload(projects[1], skills)
            if system.startswith("Fact-check"):
                return {"approved": True, "issues": []}
            return {"projects": [draft]}

    monkeypatch.setattr(full, "read_project_sources", lambda *a: {"p": {"files": []}})
    measurements = iter(
        [
            {
                "issues": ["Visible top/bottom whitespace differs"],
                "top_margin": 20,
                "bottom_margin": 60,
            },
            {"issues": [], "top_margin": 40, "bottom_margin": 40},
        ]
    )
    monkeypatch.setattr(
        full, "inspect_resume_layout", lambda *a, **kw: next(measurements)
    )
    sources = []

    def compile(source, **kw):
        sources.append(source.read_text(encoding="utf-8"))
        return {
            "resume_pdf_path": str(source.with_suffix(".pdf")),
            "pdf_pages": 1,
            "resume_text": "p DUPOON PTY LTD",
        }

    monkeypatch.setattr(module, "compile_resume_pdf", compile)
    result = module.prepare_precision_resume(
        dict(JOB, job_id="test"),
        {
            "precision_apply": {
                "enabled": True,
                "semantic_selection": True,
                "full_resume": True,
                "experience_project_id": "intern",
                "read_sources": True,
                "check_layout": True,
                "verify_claims": True,
                "catalog": str(tmp_path / "catalog.yaml"),
                "base_resume": str(base),
                "output_dir": str(tmp_path / "out"),
                "project_limit": 1,
                "bullets_per_project": 2,
            }
        },
        ai_service=AI(),
    )
    manifest = json.loads(
        Path(result["tailoring_manifest_path"]).read_text(encoding="utf-8")
    )
    assert manifest["semantic_fact_check"]["approved"]
    assert manifest["job_analysis"]["requirements"]
    assert len(sources) == 2
    assert "top=34.17pt" in sources[1]
    assert all(
        "DUPOON PTY LTD" in s and "Student visa; 485 eligible" in s for s in sources
    )


@pytest.mark.skipif(
    os.environ.get("RUN_PRECISION_LIVE_TEST") != "1",
    reason="Explicit opt-in: uses Codex allowance and real XeLaTeX",
)
def test_live_full_resume_generation(tmp_path):
    """Real source reads, AI targeting/verification and PDF layout; never applies."""
    from ronin.ai import CodexService
    from ronin.config import load_config
    from ronin.job_specific_resume import prepare_precision_resume

    config = load_config()
    config["precision_apply"]["output_dir"] = str(tmp_path / "full-resume")
    jd = {
        "job_id": "precision-qa",
        "company_name": "Example Precision QA",
        "title": "Graduate Backend Software Engineer",
        "description": (
            "This is a synthetic job description for a local integration test, "
            "not an application. Required: Python backend development, REST APIs, "
            "SQL databases, reliable asynchronous processing, automated testing "
            "and Git. Work with cloud infrastructure and CI/CD. Preferred: "
            "FastAPI, AWS, observability, streaming analytics and React interfaces. "
            "Explain engineering decisions clearly and take ownership of "
            "debugging failures."
        ),
    }
    ai = CodexService(
        default_model=config["precision_apply"]["model"], reasoning_effort="low"
    )
    try:
        result = prepare_precision_resume(jd, config, ai_service=ai)
    finally:
        ai.close()
    manifest = json.loads(
        Path(result["tailoring_manifest_path"]).read_text(encoding="utf-8")
    )
    assert not manifest["layout_check"]["issues"]
    assert manifest["semantic_fact_check"]["approved"]
    assert len(manifest["projects"]) == 3
    assert len(manifest["tailored_sections"]["experience"]["bullets"]) == 3
    assert "DUPOON PTY LTD" in result["resume_text"]
    assert "Student visa" in result["resume_text"]
    print("Verified live precision PDF:", result["resume_pdf_path"])
