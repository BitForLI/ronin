"""List and export the submitted-job ledger."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List

from rich.console import Console
from rich.table import Table

from ronin.config import get_ronin_home, load_config, load_env
from ronin.db import get_db_manager

console = Console()

_EXPORT_FIELDS = (
    "date_applied",
    "company",
    "position",
    "location",
    "source",
    "match_score",
    "selected_projects",
    "resume_file",
    "status",
    "job_url",
)


def _project_names(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return ", ".join(str(item) for item in parsed)
    except json.JSONDecodeError:
        pass
    return text


def _ledger_rows(limit: int = 500, stage: str = "") -> List[Dict[str, str]]:
    load_env()
    config = load_config()
    db = get_db_manager(config=config)
    try:
        applications = db.get_applications(limit=0)
        rows: List[Dict[str, str]] = []
        requested_stage = stage.strip().lower()
        for application in applications:
            outcome_stage = str(
                application.get("outcome_stage")
                or application.get("outcome")
                or "applied"
            ).strip()
            if requested_stage and outcome_stage.lower() != requested_stage:
                continue

            job = db.get_job_by_job_id(str(application.get("job_id") or "")) or {}
            selected = application.get("selected_projects") or job.get(
                "selected_projects"
            )
            resume_path = application.get("tailored_resume_path") or job.get(
                "tailored_resume_path"
            )
            rows.append(
                {
                    "date_applied": str(
                        application.get("date_applied")
                        or application.get("applied_at")
                        or ""
                    )[:10],
                    "company": str(application.get("company_name") or ""),
                    "position": str(
                        application.get("job_title") or application.get("title") or ""
                    ),
                    "location": str(job.get("location") or ""),
                    "source": str(application.get("source") or job.get("source") or ""),
                    "match_score": str(job.get("score") or ""),
                    "selected_projects": _project_names(selected),
                    "resume_file": str(resume_path or ""),
                    "status": outcome_stage,
                    "job_url": str(application.get("url") or job.get("url") or ""),
                }
            )
            if limit > 0 and len(rows) >= limit:
                break
        return rows
    finally:
        db.close()


def list_applications(*, limit: int = 50, stage: str = "") -> int:
    """Display submitted jobs with their tailored resume and project choices."""
    rows = _ledger_rows(limit=max(1, int(limit)), stage=stage)
    if not rows:
        console.print("[yellow]No submitted applications matched the filter.[/yellow]")
        return 0

    table = Table(title="Submitted Applications", border_style="dim")
    table.add_column("Applied", style="dim")
    table.add_column("Company", style="green")
    table.add_column("Position", style="cyan")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Projects")
    table.add_column("Status")
    for row in rows:
        table.add_row(
            row["date_applied"],
            row["company"][:25],
            row["position"][:42],
            row["match_score"],
            row["selected_projects"][:38],
            row["status"],
        )
    console.print(table)
    return 0


def _write_csv(rows: Iterable[Dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_EXPORT_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_cell(value: str) -> str:
    return str(value or "").replace("|", r"\|").replace("\n", " ")


def _write_markdown(rows: Iterable[Dict[str, str]], path: Path) -> None:
    rows = list(rows)
    headings = [field.replace("_", " ").title() for field in _EXPORT_FIELDS]
    lines = [
        "# Submitted Applications",
        "",
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join("---" for _ in headings) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row[field]) for field in _EXPORT_FIELDS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_applications(
    *,
    output: str = "",
    file_format: str = "csv",
    stage: str = "",
    limit: int = 0,
) -> int:
    """Export the submitted-job ledger to CSV or Markdown."""
    resolved_format = file_format.strip().lower()
    if resolved_format not in {"csv", "markdown", "md"}:
        console.print("[red]Format must be csv or markdown.[/red]")
        return 1
    suffix = ".csv" if resolved_format == "csv" else ".md"
    destination = (
        Path(output).expanduser().resolve()
        if output
        else (
            get_ronin_home()
            / "exports"
            / f"applications-{date.today().isoformat()}{suffix}"
        ).resolve()
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = _ledger_rows(limit=max(0, int(limit)), stage=stage)
    if resolved_format == "csv":
        _write_csv(rows, destination)
    else:
        _write_markdown(rows, destination)
    console.print(
        f"[green]Exported {len(rows)} submitted application(s):[/green] {destination}"
    )
    return 0
