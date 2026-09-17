"""CLI operations for repository-backed, job-specific resume generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.table import Table

from ronin.config import get_ronin_home, load_config, load_env
from ronin.db import get_db_manager
from ronin.job_specific_resume import (
    JobSpecificResumeError,
    catalog_to_prompt_preview,
    load_project_catalog,
    load_tailoring_rules,
    prepare_precision_resume,
    select_projects,
    tailor_projects_with_ai,
    write_project_catalog_draft,
    write_tailoring_artifacts,
)
from ronin.profile import load_profile

console = Console()


def prepare_application_resume(
    record: Dict[str, Any], db: Any, config: Dict[str, Any], *, ai_service: Any = None
) -> Dict[str, str]:
    """Save the exact tailored PDF metadata before allowing an application."""
    prepared = prepare_precision_resume(record, config, ai_service=ai_service)
    if not prepared:
        return {}
    metadata = {
        key: prepared[key]
        for key in (
            "selected_projects",
            "tailored_resume_path",
            "tailoring_manifest_path",
            "tailoring_generated_at",
        )
    }
    if not db.update_record(int(record["id"]), metadata):
        raise JobSpecificResumeError(
            "Tailored PDF generated but metadata could not be saved; application stopped"
        )
    record.update(metadata)
    console.print(f"[cyan]Job-specific PDF:[/cyan] {prepared['resume_pdf_path']}")
    return {
        "resume_pdf_path": prepared["resume_pdf_path"],
        "resume_text": prepared["resume_text"],
    }


def index_projects(*, root: str, output: str = "") -> int:
    """Create a review-required project fact catalog from local repositories."""
    source_root = Path(root).expanduser().resolve()
    destination = (
        Path(output).expanduser().resolve()
        if output
        else (get_ronin_home() / "projects.yaml").resolve()
    )
    try:
        path = write_project_catalog_draft(source_root, destination)
    except JobSpecificResumeError as exc:
        console.print(f"[red]Project indexing failed:[/red] {exc}")
        return 1

    console.print(f"[green]Project catalog draft written:[/green] {path}")
    console.print(
        "[yellow]Review required:[/yellow] add source-backed STAR evidence and set "
        "[bold]needs_review: false[/bold] only for verified projects."
    )
    return 0


def _load_job(
    *,
    job_id: str,
    jd_file: str,
    title: str,
    company: str,
) -> Tuple[Dict[str, Any], Optional[Any], Optional[Dict[str, Any]]]:
    if bool(job_id) == bool(jd_file):
        raise JobSpecificResumeError("Provide exactly one of --job-id or --jd-file")

    if jd_file:
        path = Path(jd_file).expanduser().resolve()
        if not path.exists():
            raise JobSpecificResumeError(f"Job description file not found: {path}")
        description = path.read_text(encoding="utf-8")
        return (
            {
                "job_id": path.stem,
                "title": title or path.stem.replace("_", " ").replace("-", " "),
                "company_name": company,
                "description": description,
            },
            None,
            None,
        )

    db = get_db_manager()
    record = db.get_job_by_job_id(job_id)
    if not record:
        db.close()
        raise JobSpecificResumeError(f"Job not found in Ronin database: {job_id}")
    return record, db, record


def _ai_writer(provider: str) -> Any:
    if provider in {"codex", "openai", "anthropic"}:
        from ronin.ai import CodexService

        return CodexService(default_model="gpt-5.6-terra", reasoning_effort="low")
    raise JobSpecificResumeError(f"Unsupported AI provider {provider!r}; use codex")


def _show_matches(matches: list[Dict[str, Any]]) -> None:
    table = Table(title="Selected GitHub Projects", border_style="dim")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Project", style="cyan")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Why it matches")
    table.add_column("Verified", justify="center")
    for index, match in enumerate(matches, start=1):
        table.add_row(
            str(index),
            str(match["name"]),
            f"{float(match['score']):.1f}",
            str(match["reason"])[:80],
            "no" if match["needs_review"] else "yes",
        )
    console.print(table)


def build_resume(
    *,
    job_id: str = "",
    jd_file: str = "",
    title: str = "",
    company: str = "",
    catalog: str = "",
    base_resume: str = "",
    output_dir: str = "",
    project_limit: int = 3,
    min_project_score: float = 5.0,
    bullets_per_project: int = 3,
    max_words_per_bullet: int = 42,
    provider: str = "",
    model: str = "",
    preview: bool = False,
    pdf: bool = False,
) -> int:
    """Select projects, write validated STAR bullets and create resume artifacts."""
    load_env()
    db = None
    try:
        job, db, db_record = _load_job(
            job_id=job_id,
            jd_file=jd_file,
            title=title,
            company=company,
        )
        catalog_path = (
            Path(catalog).expanduser().resolve()
            if catalog
            else (get_ronin_home() / "projects.yaml").resolve()
        )
        projects = load_project_catalog(catalog_path)
        matches = select_projects(
            projects,
            job_title=str(job.get("title") or title),
            job_description=str(job.get("description") or ""),
            limit=int(project_limit),
            min_score=float(min_project_score),
        )
        _show_matches(catalog_to_prompt_preview(matches))
        if preview:
            console.print(
                "[dim]Preview only. No AI call, resume write or database update was made.[/dim]"
            )
            return 0

        tailoring_rules = load_tailoring_rules(load_config())
        profile = load_profile()
        resolved_provider = (provider or profile.ai.analysis_provider).strip().lower()
        resolved_model = (model or profile.ai.analysis_model).strip()
        writer = _ai_writer(resolved_provider)
        result = tailor_projects_with_ai(
            ai_service=writer,
            model=resolved_model,
            job_id=str(job.get("job_id") or job_id),
            job_title=str(job.get("title") or title),
            company=str(job.get("company_name") or company),
            job_description=str(job.get("description") or ""),
            matches=matches,
            bullets_per_project=int(bullets_per_project),
            max_words_per_bullet=int(max_words_per_bullet),
            tailoring_rules=tailoring_rules,
        )

        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else (
                get_ronin_home()
                / "tailored_resumes"
                / str(job.get("job_id") or job_id or "manual")
            ).resolve()
        )
        artifacts = write_tailoring_artifacts(
            result=result,
            output_dir=destination,
            base_resume=(Path(base_resume) if base_resume else None),
        )
        if pdf:
            from ronin.config import load_config
            from ronin.job_specific_resume import compile_resume_pdf

            if not base_resume or Path(base_resume).suffix.lower() != ".tex":
                raise JobSpecificResumeError(
                    "--pdf requires --base-resume with a LaTeX template"
                )
            settings = load_config().get("precision_apply", {})
            compiled = compile_resume_pdf(
                Path(artifacts["resume_path"]),
                command=settings.get("compiler_command", ()),
                max_pages=int(settings.get("max_pages", 1)),
            )
            manifest = json.loads(
                Path(artifacts["manifest_path"]).read_text(encoding="utf-8")
            )
            manifest.update(
                pdf_path=compiled["resume_pdf_path"], pdf_pages=compiled["pdf_pages"]
            )
            Path(artifacts["manifest_path"]).write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            artifacts["resume_path"] = compiled["resume_pdf_path"]

        selected_ids = [project.project_id for project in result.projects]
        generated_at = datetime.now(timezone.utc).isoformat()
        if db is not None and db_record is not None:
            updated = db.update_record(
                int(db_record["id"]),
                {
                    "selected_projects": json.dumps(selected_ids),
                    "tailored_resume_path": artifacts["resume_path"],
                    "tailoring_manifest_path": artifacts["manifest_path"],
                    "tailoring_generated_at": generated_at,
                },
            )
            if not updated:
                raise JobSpecificResumeError(
                    "Resume was written, but job tailoring metadata was not saved"
                )

        console.print(
            f"[green]Tailored resume written:[/green] {artifacts['resume_path']}"
        )
        console.print(
            f"[green]Evidence manifest written:[/green] {artifacts['manifest_path']}"
        )
        console.print(
            "[yellow]Review before submission.[/yellow] The manifest shows why each "
            "project was selected and which repository facts support every bullet."
        )
        return 0
    except (JobSpecificResumeError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Tailoring failed:[/red] {exc}")
        return 1
    finally:
        if db is not None:
            db.close()
