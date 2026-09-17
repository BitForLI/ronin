"""Build a job-specific projects section from verified repository facts.

This module deliberately separates selection from writing:

1. A deterministic matcher ranks projects against the job description.
2. An AI writer receives only the selected projects and their evidence facts.
3. A validator rejects unsupported projects, technologies, evidence references,
   and numeric claims before any resume artifact is written.

The result is not a keyword-swapped resume. The projects section is composed
again for each job while the underlying claims remain traceable to source files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


class JobSpecificResumeError(RuntimeError):
    """A job-specific resume could not pass a hard validation gate."""


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "to",
        "using",
        "we",
        "will",
        "with",
        "you",
        "your",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-/]{1,}")
_NUMBER_RE = re.compile(
    r"\$\s?[\d.,]+\s?[kmb]?\+?"
    r"|\b\d[\d,.]*\s?%"
    r"|\b\d+(?:\.\d+)?x\b"
    r"|\b\d{2,}(?:[,.]\d+)*\b",
    re.IGNORECASE,
)

_TECH_ALIASES = {
    "amazon web services": "aws",
    "c sharp": "c#",
    "c-sharp": "c#",
    "dotnet": ".net",
    "github actions": "github actions",
    "google cloud platform": "gcp",
    "javascript": "javascript",
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "postgres": "postgresql",
    "react.js": "react",
    "reactjs": "react",
    "typescript": "typescript",
}

_ACTION_KINDS = frozenset({"action", "implementation"})
_CONTEXT_KINDS = frozenset({"situation", "task", "context", "problem"})
_RESULT_KINDS = frozenset({"result", "outcome", "impact"})
_ACTION_VERBS = frozenset(
    {
        "added",
        "architected",
        "automated",
        "built",
        "configured",
        "created",
        "delivered",
        "deployed",
        "designed",
        "developed",
        "documented",
        "engineered",
        "established",
        "implemented",
        "improved",
        "integrated",
        "instrumented",
        "led",
        "optimised",
        "optimized",
        "reduced",
        "refactored",
        "tested",
        "validated",
    }
)

_EXTENSION_TECHNOLOGIES = {
    ".cs": "C#",
    ".css": "CSS",
    ".dart": "Dart",
    ".go": "Go",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "React",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".sql": "SQL",
    ".swift": "Swift",
    ".tf": "Terraform",
    ".ts": "TypeScript",
    ".tsx": "React",
}

_MANIFEST_TECHNOLOGIES = {
    "asp.net": "ASP.NET Core",
    "airflow": "Apache Airflow",
    "aws": "AWS",
    "azure": "Azure",
    "clickhouse": "ClickHouse",
    "docker": "Docker",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "flutter": "Flutter",
    "github actions": "GitHub Actions",
    "graphql": "GraphQL",
    "kafka": "Apache Kafka",
    "langgraph": "LangGraph",
    "langchain": "LangChain",
    "mcp": "MCP",
    "mysql": "MySQL",
    "next": "Next.js",
    "node": "Node.js",
    "postgres": "PostgreSQL",
    "pytest": "pytest",
    "react": "React",
    "redis": "Redis",
    "sqlite": "SQLite",
    "spring": "Spring Boot",
    "tailwind": "Tailwind CSS",
    "terraform": "Terraform",
    "vite": "Vite",
}

_IGNORED_REPOSITORY_PARTS = frozenset(
    {
        ".git",
        ".idea",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        ".venv",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalise_term(value: str) -> str:
    term = _clean_text(value).lower()
    return _TECH_ALIASES.get(term, term)


def _tokens(value: str) -> set[str]:
    return {
        _TECH_ALIASES.get(token.lower(), token.lower())
        for token in _TOKEN_RE.findall(value or "")
        if token.lower() not in _STOP_WORDS
    }


def _numeric_claims(value: str) -> set[str]:
    claims = set()
    for match in _NUMBER_RE.findall(value or ""):
        claims.add(re.sub(r"[\s,]", "", match).lower().rstrip("+"))
    return claims


def _string_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise JobSpecificResumeError("Expected a string or list of strings")
    cleaned = tuple(text for item in value if (text := _clean_text(item)))
    return cleaned


@dataclass(frozen=True)
class EvidenceFact:
    """One source-backed fact that the resume writer may use."""

    id: str
    kind: str
    text: str
    source: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], fallback_id: str) -> "EvidenceFact":
        text = _clean_text(raw.get("text"))
        if not text:
            raise JobSpecificResumeError(f"Evidence {fallback_id!r} has no text")
        return cls(
            id=_clean_text(raw.get("id")) or fallback_id,
            kind=_clean_text(raw.get("kind")).lower() or "context",
            text=text,
            source=_clean_text(raw.get("source")),
        )


@dataclass(frozen=True)
class ProjectFact:
    """Verified facts for one portfolio project."""

    id: str
    name: str
    repository: str = ""
    summary: str = ""
    technologies: Tuple[str, ...] = ()
    capabilities: Tuple[str, ...] = ()
    domains: Tuple[str, ...] = ()
    evidence: Tuple[EvidenceFact, ...] = ()
    needs_review: bool = False
    included_in_experience: bool = False

    @property
    def evidence_by_id(self) -> Dict[str, EvidenceFact]:
        return {fact.id: fact for fact in self.evidence}

    @property
    def searchable_text(self) -> str:
        parts: List[str] = [self.name, self.summary]
        parts.extend(self.technologies)
        parts.extend(self.capabilities)
        parts.extend(self.domains)
        parts.extend(fact.text for fact in self.evidence)
        return " ".join(parts)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], index: int) -> "ProjectFact":
        project_id = _clean_text(raw.get("id"))
        name = _clean_text(raw.get("name"))
        if not project_id or not name:
            raise JobSpecificResumeError(
                f"Project #{index} requires non-empty 'id' and 'name'"
            )

        facts: List[EvidenceFact] = []
        raw_facts = raw.get("evidence") or raw.get("facts") or []
        if not isinstance(raw_facts, list):
            raise JobSpecificResumeError(
                f"Project {project_id!r} evidence must be a list"
            )
        for fact_index, item in enumerate(raw_facts, start=1):
            if isinstance(item, str):
                item = {"text": item, "kind": "context"}
            if not isinstance(item, dict):
                raise JobSpecificResumeError(
                    f"Project {project_id!r} evidence #{fact_index} is invalid"
                )
            facts.append(EvidenceFact.from_dict(item, fallback_id=f"fact:{fact_index}"))

        # Accept concise STAR-shaped YAML as a convenience, then normalise it
        # into the same evidence model used by validation.
        convenience_fields = (
            ("situation", "situation"),
            ("task", "task"),
            ("actions", "action"),
            ("results", "result"),
        )
        for field_name, kind in convenience_fields:
            for item_index, text in enumerate(
                _string_list(raw.get(field_name)), start=1
            ):
                facts.append(
                    EvidenceFact(
                        id=f"{kind}:{item_index}",
                        kind=kind,
                        text=text,
                        source=_clean_text(raw.get("source")),
                    )
                )

        seen_ids: set[str] = set()
        for fact in facts:
            if fact.id in seen_ids:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} repeats evidence id {fact.id!r}"
                )
            seen_ids.add(fact.id)

        return cls(
            id=project_id,
            name=name,
            repository=_clean_text(raw.get("repository") or raw.get("github")),
            summary=_clean_text(raw.get("summary")),
            technologies=_string_list(raw.get("technologies")),
            capabilities=_string_list(raw.get("capabilities")),
            domains=_string_list(raw.get("domains")),
            evidence=tuple(facts),
            needs_review=bool(raw.get("needs_review", False)),
            included_in_experience=bool(raw.get("included_in_experience", False)),
        )


@dataclass(frozen=True)
class ProjectMatch:
    """A project's deterministic relevance score for one job."""

    project: ProjectFact
    score: float
    matched_technologies: Tuple[str, ...] = ()
    matched_capabilities: Tuple[str, ...] = ()
    matched_domains: Tuple[str, ...] = ()
    matched_terms: Tuple[str, ...] = ()

    def reason(self) -> str:
        signals = list(self.matched_technologies)
        signals.extend(self.matched_capabilities)
        signals.extend(self.matched_domains)
        signals.extend(self.matched_terms[:4])
        return ", ".join(dict.fromkeys(signals)) or "general project overlap"


@dataclass(frozen=True)
class TailoredProject:
    """Validated resume copy for one selected project."""

    project_id: str
    name: str
    repository: str
    technologies: Tuple[str, ...]
    bullets: Tuple[str, ...]
    evidence_ids: Tuple[Tuple[str, ...], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TailoringResult:
    """A validated set of project sections for one job."""

    job_id: str
    job_title: str
    company: str
    projects: Tuple[TailoredProject, ...]
    matches: Tuple[ProjectMatch, ...]


def load_project_catalog(path: Path) -> List[ProjectFact]:
    """Load and validate a project fact catalog from YAML."""
    source = Path(path).expanduser()
    if not source.exists():
        raise JobSpecificResumeError(f"Project catalog not found: {source}")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw_projects = raw.get("projects")
    else:
        raw_projects = raw
    if not isinstance(raw_projects, list) or not raw_projects:
        raise JobSpecificResumeError("Project catalog must contain a non-empty list")
    projects = [
        ProjectFact.from_dict(item, index=index)
        for index, item in enumerate(raw_projects, start=1)
        if isinstance(item, dict)
    ]
    if len(projects) != len(raw_projects):
        raise JobSpecificResumeError("Every project catalog entry must be a mapping")
    ids = [project.id for project in projects]
    if len(ids) != len(set(ids)):
        raise JobSpecificResumeError("Project ids must be unique")
    return projects


def _read_small_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        if path.stat().st_size > max_chars:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _first_readme(repository: Path) -> Optional[Path]:
    for name in ("README.md", "README.MD", "README.rst", "README.txt"):
        candidate = repository / name
        if candidate.exists():
            return candidate
    return None


def _readme_summary(text: str) -> str:
    paragraphs: List[str] = []
    current: List[str] = []
    in_code = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or line.startswith("#") or line.startswith("!"):
            continue
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith(("- ", "* ", "|", ">")):
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    for paragraph in paragraphs:
        cleaned = _clean_text(re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", paragraph))
        if len(cleaned) >= 30:
            return cleaned[:500]
    return ""


def _readme_title(text: str, fallback: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("# "):
            continue
        title = _clean_text(re.sub(r"[`*_]", "", line[2:]))
        if title and len(title) <= 80:
            return title
    return fallback


def _repository_url(repository: Path) -> str:
    config_path = repository / ".git" / "config"
    text = _read_small_text(config_path, max_chars=50_000)
    match = re.search(
        r'\[remote\s+"origin"\][\s\S]*?url\s*=\s*(\S+)', text, re.IGNORECASE
    )
    if not match:
        return ""
    url = match.group(1).strip()
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.split(":", 1)[1]
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _documentation_evidence(
    repository: Path, readme: Optional[Path]
) -> List[Dict[str, str]]:
    candidates: List[Path] = []
    resume_bullets = repository / "docs" / "resume-bullets.md"
    if resume_bullets.exists():
        candidates.append(resume_bullets)
    if readme:
        candidates.append(readme)

    evidence: List[Dict[str, str]] = []
    seen: set[str] = set()

    def add_evidence(text: str, kind: str, path: Path) -> None:
        text = _clean_text(re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", text))
        if len(text) < 25 or len(text) > 500 or text.lower() in seen:
            return
        first_word_match = re.match(r"[A-Za-z]+", text)
        first_word = first_word_match.group(0).lower() if first_word_match else ""
        resolved_kind = kind
        if first_word in _ACTION_VERBS or first_word.endswith("ed"):
            resolved_kind = "action"
        seen.add(text.lower())
        evidence.append(
            {
                "id": f"draft:{len(evidence) + 1}",
                "kind": resolved_kind,
                "text": text,
                "source": str(path.relative_to(repository)).replace("\\", "/"),
            }
        )

    for path in candidates:
        current_kind = "context"
        pending_bullet = ""
        for raw in [*_read_small_text(path).splitlines(), ""]:
            line = raw.strip()
            if line.startswith("#"):
                if pending_bullet:
                    add_evidence(pending_bullet, current_kind, path)
                    pending_bullet = ""
                heading = line.lstrip("#").strip().lower()
                if any(word in heading for word in ("result", "outcome", "impact")):
                    current_kind = "result"
                elif any(
                    word in heading
                    for word in ("implementation", "built", "feature", "work")
                ):
                    current_kind = "action"
                elif any(word in heading for word in ("problem", "goal", "context")):
                    current_kind = "context"
                continue
            if line.startswith(("- ", "* ")):
                if pending_bullet:
                    add_evidence(pending_bullet, current_kind, path)
                    if len(evidence) >= 12:
                        return evidence
                pending_bullet = line[2:]
                continue
            if pending_bullet and line and not line.startswith(("```", "|")):
                pending_bullet += " " + line
                continue
            if pending_bullet:
                add_evidence(pending_bullet, current_kind, path)
                pending_bullet = ""
                if len(evidence) >= 12:
                    return evidence
    return evidence


def index_local_projects(root: Path) -> List[Dict[str, Any]]:
    """Create a review-required project catalog draft from local repositories.

    The indexer only extracts observable repository signals. It intentionally
    does not infer STAR results or claim that README marketing copy is true.
    Users must review the generated YAML and set ``needs_review: false`` before
    a project is eligible for automatic resume generation.
    """
    source_root = Path(root).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise JobSpecificResumeError(f"Repository root not found: {source_root}")

    if (source_root / ".git").exists():
        repositories = [source_root]
    else:
        repositories = sorted(
            [
                child
                for child in source_root.iterdir()
                if child.is_dir()
                and ((child / ".git").exists() or _first_readme(child))
            ],
            key=lambda path: path.name.lower(),
        )
    if not repositories:
        raise JobSpecificResumeError(f"No repositories found under: {source_root}")

    projects: List[Dict[str, Any]] = []
    for repository in repositories:
        technologies: set[str] = set()
        searchable_manifest = ""
        found_dockerfile = False
        inspected = 0
        for path in repository.rglob("*"):
            try:
                relative = path.relative_to(repository)
            except ValueError:
                continue
            if any(part in _IGNORED_REPOSITORY_PARTS for part in relative.parts):
                continue
            if not path.is_file():
                continue
            inspected += 1
            if inspected > 5_000:
                break
            technology = _EXTENSION_TECHNOLOGIES.get(path.suffix.lower())
            if technology:
                technologies.add(technology)
            if path.name.lower() in {"dockerfile", "docker-compose.yml", "compose.yml"}:
                found_dockerfile = True
            if path.name.lower() in {
                "dockerfile",
                "package.json",
                "pyproject.toml",
                "requirements.txt",
                "requirements-dev.txt",
                "pom.xml",
                "build.gradle",
            }:
                searchable_manifest += (
                    "\n" + _read_small_text(path, max_chars=120_000).lower()
                )
            if ".github" in relative.parts and "workflows" in relative.parts:
                technologies.add("GitHub Actions")

        readme = _first_readme(repository)
        readme_text = _read_small_text(readme) if readme else ""
        searchable_manifest += "\n" + readme_text.lower()
        for signal, technology in _MANIFEST_TECHNOLOGIES.items():
            if signal in searchable_manifest:
                technologies.add(technology)
        if found_dockerfile:
            technologies.add("Docker")
        summary = _readme_summary(readme_text)
        evidence = _documentation_evidence(repository, readme)
        if summary:
            evidence.insert(
                0,
                {
                    "id": "draft:summary",
                    "kind": "context",
                    "text": summary,
                    "source": readme.name if readme else "",
                },
            )

        project_id = _slug(repository.name)
        projects.append(
            {
                "id": project_id,
                "name": _readme_title(readme_text, repository.name),
                "repository": _repository_url(repository),
                "summary": summary,
                "technologies": sorted(technologies, key=str.lower),
                "capabilities": [],
                "domains": [],
                "evidence": evidence,
                "needs_review": True,
            }
        )
    return projects


def write_project_catalog_draft(root: Path, output_path: Path) -> Path:
    """Index local repositories and write an English review-required YAML draft."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "instructions": (
            "Review every project, add capabilities/domains and source-backed "
            "situation, task, action and result evidence, then set needs_review "
            "to false. Automatic tailoring excludes unreviewed projects."
        ),
        "projects": index_local_projects(root),
    }
    destination.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=False,
            width=100,
        ),
        encoding="utf-8",
    )
    return destination


def _phrase_present(term: str, text: str, text_tokens: set[str]) -> bool:
    normalised = _normalise_term(term)
    if not normalised:
        return False
    if " " in normalised or any(char in normalised for char in ".+#"):
        aliases = {normalised, _TECH_ALIASES.get(normalised, normalised)}
        return any(alias in text for alias in aliases)
    return normalised in text_tokens


def score_project(project: ProjectFact, job_text: str) -> ProjectMatch:
    """Score one project against a job description on a 0-100 scale."""
    normalised_job = _clean_text(job_text).lower()
    job_tokens = _tokens(normalised_job)

    technologies = tuple(
        tech
        for tech in project.technologies
        if _phrase_present(tech, normalised_job, job_tokens)
    )
    capabilities = tuple(
        item
        for item in project.capabilities
        if _phrase_present(item, normalised_job, job_tokens)
        or bool(_tokens(item) & job_tokens)
    )
    domains = tuple(
        item
        for item in project.domains
        if _phrase_present(item, normalised_job, job_tokens)
        or bool(_tokens(item) & job_tokens)
    )

    project_tokens = _tokens(project.searchable_text)
    overlap = sorted(project_tokens & job_tokens)
    overlap_ratio = len(overlap) / max(1, len(job_tokens))

    score = min(45.0, len(technologies) * 9.0)
    score += min(30.0, len(capabilities) * 6.0)
    score += min(10.0, len(domains) * 5.0)
    score += min(15.0, overlap_ratio * 100.0)

    return ProjectMatch(
        project=project,
        score=round(min(100.0, score), 2),
        matched_technologies=technologies,
        matched_capabilities=capabilities,
        matched_domains=domains,
        matched_terms=tuple(overlap),
    )


def select_projects(
    projects: Sequence[ProjectFact],
    job_title: str,
    job_description: str,
    limit: int = 3,
    min_score: float = 5.0,
    allow_unreviewed: bool = False,
) -> List[ProjectMatch]:
    """Return the strongest source-backed projects for a job."""
    if limit < 1 or limit > 5:
        raise JobSpecificResumeError("Project limit must be between 1 and 5")
    job_text = f"{job_title}\n{job_description}".strip()
    if not job_text:
        raise JobSpecificResumeError("Job title or description is required")

    eligible = [
        project
        for project in projects
        if not project.included_in_experience
        and (allow_unreviewed or not project.needs_review)
    ]
    if not eligible:
        raise JobSpecificResumeError(
            "No reviewed projects are available outside existing work experience; "
            "review the catalog without duplicating internship work"
        )
    ranked = [score_project(project, job_text) for project in eligible]
    ranked.sort(key=lambda match: (-match.score, match.project.name.lower()))
    selected = [match for match in ranked if match.score >= min_score][:limit]
    if not selected:
        raise JobSpecificResumeError(
            "No project met the minimum relevance score; refusing to add filler"
        )
    return selected


def build_tailoring_prompts(
    *,
    job_title: str,
    company: str,
    job_description: str,
    matches: Sequence[ProjectMatch],
    bullets_per_project: int = 3,
    max_words_per_bullet: int = 42,
    tailoring_rules: str = "",
) -> Tuple[str, str]:
    """Build the constrained AI prompts for job-specific STAR writing."""
    if bullets_per_project not in {2, 3}:
        raise JobSpecificResumeError("Bullets per project must be 2 or 3")

    projects_payload = []
    for match in matches:
        project = match.project
        projects_payload.append(
            {
                "id": project.id,
                "name": project.name,
                "repository": project.repository,
                "summary": project.summary,
                "verified_technologies": list(project.technologies),
                "verified_capabilities": list(project.capabilities),
                "verified_domains": list(project.domains),
                "match_score": match.score,
                "match_reason": match.reason(),
                "evidence": [
                    {
                        "id": fact.id,
                        "kind": fact.kind,
                        "text": fact.text,
                        "source": fact.source,
                    }
                    for fact in project.evidence
                ],
            }
        )

    system_prompt = f"""
You write the Technical Projects section of a software resume for one job.
The projects have already been selected. Rewrite each project from its verified
facts using compressed STAR structure: establish the problem or task, describe
the candidate's concrete engineering action, and finish with a defensible result.

Hard rules:
- Use every supplied project exactly once and preserve its project id and name.
- Produce exactly {bullets_per_project} bullets per project.
- Every bullet must be at most {max_words_per_bullet} words and should normally
  render as two or three resume lines, never a dangling fragment.
- Begin with a specific past-tense action verb.
- Use only supplied facts and verified technologies. Never invent a feature,
  metric, user count, performance result, deployment state, or business impact.
- Each bullet must list the evidence ids that support it. Reference at least one
  action/implementation fact and one situation/task/result/context fact.
- Prefer job-relevant facts, but do not repeat the job description as a claim.
- Technology Stack must be a list, not a descriptive phrase.
- Choose at most six job-relevant technologies for each project header; do not
  list every dependency. Preserve the candidate's contribution, not just tasks.
- Use direct, natural Australian English. No first person, em dashes, hype, or
  generic claims such as 'passionate', 'cutting-edge', or 'leveraged my skills'.

Return one JSON object with this exact shape:
{{
  "projects": [
    {{
      "project_id": "catalog id",
      "name": "catalog name",
      "technologies": ["verified technology"],
      "bullets": [
        {{"text": "STAR bullet", "evidence_ids": ["fact id"]}}
      ]
    }}
  ]
}}
""".strip()

    if tailoring_rules.strip():
        system_prompt += (
            "\n\nActive resume tailoring policy:\n"
            + tailoring_rules.strip()
            + "\nApply relevant writing rules while returning only the JSON schema "
            "above. This operation rewrites Technical Projects only; do not "
            "rewrite other sections or claim an application was submitted."
        )

    user_prompt = (
        f"Job title: {job_title}\n"
        f"Company: {company}\n\n"
        f"Job description:\n{job_description.strip()}\n\n"
        "Selected project facts:\n"
        + json.dumps(projects_payload, indent=2, ensure_ascii=False)
    )
    return system_prompt, user_prompt


def validate_tailored_projects(
    payload: Dict[str, Any],
    matches: Sequence[ProjectMatch],
    *,
    bullets_per_project: int = 3,
    max_words_per_bullet: int = 42,
) -> Tuple[TailoredProject, ...]:
    """Validate AI output against selected projects and evidence facts."""
    raw_projects = payload.get("projects") if isinstance(payload, dict) else None
    if not isinstance(raw_projects, list):
        raise JobSpecificResumeError("AI output has no projects list")

    selected = {match.project.id: match.project for match in matches}
    output_ids = [_clean_text(item.get("project_id")) for item in raw_projects]
    if set(output_ids) != set(selected) or len(output_ids) != len(selected):
        raise JobSpecificResumeError(
            "AI output project set differs from the selected project set"
        )

    validated: List[TailoredProject] = []
    for item in raw_projects:
        if not isinstance(item, dict):
            raise JobSpecificResumeError("AI project output must be an object")
        project_id = _clean_text(item.get("project_id"))
        project = selected[project_id]
        if _clean_text(item.get("name")) != project.name:
            raise JobSpecificResumeError(
                f"AI changed the name of project {project_id!r}"
            )

        raw_technologies = item.get("technologies") or []
        if not isinstance(raw_technologies, list) or not raw_technologies:
            raise JobSpecificResumeError(
                f"Project {project_id!r} technologies must be a non-empty list"
            )
        verified_tech = {
            _normalise_term(value): value for value in project.technologies
        }
        technologies: List[str] = []
        for value in raw_technologies:
            normalised = _normalise_term(str(value))
            if normalised not in verified_tech:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} contains unverified technology {value!r}"
                )
            technologies.append(verified_tech[normalised])

        raw_bullets = item.get("bullets") or []
        if not isinstance(raw_bullets, list) or len(raw_bullets) != bullets_per_project:
            raise JobSpecificResumeError(
                f"Project {project_id!r} must contain exactly "
                f"{bullets_per_project} bullets"
            )

        evidence_map = project.evidence_by_id
        bullets: List[str] = []
        bullet_evidence: List[Tuple[str, ...]] = []
        for bullet_index, raw_bullet in enumerate(raw_bullets, start=1):
            if not isinstance(raw_bullet, dict):
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} is invalid"
                )
            text = _clean_text(raw_bullet.get("text"))
            evidence_ids = _string_list(raw_bullet.get("evidence_ids"))
            if not text or not evidence_ids:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} lacks text or evidence"
                )
            if len(text.split()) > max_words_per_bullet:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} exceeds "
                    f"{max_words_per_bullet} words"
                )
            first_word_match = re.match(r"[A-Za-z]+", text)
            first_word = first_word_match.group(0).lower() if first_word_match else ""
            if first_word not in _ACTION_VERBS and not first_word.endswith("ed"):
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} must begin "
                    "with a past-tense action verb"
                )
            unknown = [
                fact_id for fact_id in evidence_ids if fact_id not in evidence_map
            ]
            if unknown:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} cites unknown "
                    f"evidence: {', '.join(unknown)}"
                )

            kinds = {evidence_map[fact_id].kind for fact_id in evidence_ids}
            if not kinds & _ACTION_KINDS:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} has no action evidence"
                )
            if not kinds & (_CONTEXT_KINDS | _RESULT_KINDS):
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} has no STAR context/result evidence"
                )

            evidence_text = " ".join(
                evidence_map[fact_id].text for fact_id in evidence_ids
            )
            invented_numbers = _numeric_claims(text) - _numeric_claims(evidence_text)
            if invented_numbers:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} contains unsupported "
                    f"numeric claims: {', '.join(sorted(invented_numbers))}"
                )
            if "—" in text:
                raise JobSpecificResumeError(
                    f"Project {project_id!r} bullet {bullet_index} contains an em dash"
                )

            bullets.append(text)
            bullet_evidence.append(evidence_ids)

        validated.append(
            TailoredProject(
                project_id=project.id,
                name=project.name,
                repository=project.repository,
                technologies=tuple(dict.fromkeys(technologies)),
                bullets=tuple(bullets),
                evidence_ids=tuple(bullet_evidence),
            )
        )
    return tuple(validated)


def tailor_projects_with_ai(
    *,
    ai_service: Any,
    model: str,
    job_id: str,
    job_title: str,
    company: str,
    job_description: str,
    matches: Sequence[ProjectMatch],
    bullets_per_project: int = 3,
    max_words_per_bullet: int = 42,
    max_attempts: int = 2,
    tailoring_rules: str = "",
) -> TailoringResult:
    """Generate and validate a job-specific set of project sections."""
    system_prompt, user_prompt = build_tailoring_prompts(
        job_title=job_title,
        company=company,
        job_description=job_description,
        matches=matches,
        bullets_per_project=bullets_per_project,
        max_words_per_bullet=max_words_per_bullet,
        tailoring_rules=tailoring_rules,
    )
    if max_attempts < 1 or max_attempts > 3:
        raise JobSpecificResumeError("AI generation attempts must be between 1 and 3")

    projects: Optional[Tuple[TailoredProject, ...]] = None
    last_error = "AI writer returned no usable JSON"
    current_prompt = user_prompt
    for attempt in range(1, max_attempts + 1):
        payload = ai_service.chat_completion(
            system_prompt=system_prompt,
            user_message=current_prompt,
            model=model,
            temperature=0.2,
        )
        if not isinstance(payload, dict):
            last_error = "AI writer returned no usable JSON"
        else:
            try:
                projects = validate_tailored_projects(
                    payload,
                    matches,
                    bullets_per_project=bullets_per_project,
                    max_words_per_bullet=max_words_per_bullet,
                )
                break
            except JobSpecificResumeError as exc:
                last_error = str(exc)
        if attempt < max_attempts:
            current_prompt = (
                user_prompt
                + "\n\nThe previous draft was rejected by the fact validator: "
                + last_error
                + "\nReturn a corrected JSON object that follows every hard rule."
            )
    if projects is None:
        raise JobSpecificResumeError(
            f"AI draft failed validation after {max_attempts} attempt(s): {last_error}"
        )
    return TailoringResult(
        job_id=_clean_text(job_id) or "manual-job",
        job_title=_clean_text(job_title),
        company=_clean_text(company),
        projects=projects,
        matches=tuple(matches),
    )


def render_markdown_projects(result: TailoringResult) -> str:
    """Render a validated projects section as Markdown."""
    lines = ["## Technical Projects", ""]
    for project in result.projects:
        name = project.name
        if project.repository:
            name = f"[{name}]({project.repository})"
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"Technology Stack: {', '.join(project.technologies)}")
        lines.append("")
        lines.extend(f"- {bullet}" for bullet in project.bullets)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_latex_projects(result: TailoringResult) -> str:
    """Render a validated projects section for Ronin's LaTeX resume style."""
    lines = [r"\section{Technical Projects}", "", r"\begin{cvblocks}"]
    for project in result.projects:
        name = _escape_latex(project.name)
        if project.repository:
            project_link = _escape_latex(project.repository)
            heading = rf"\textbf{{\href{{{project_link}}}{{{name}}}}}"
        else:
            heading = rf"\textbf{{{name}}}"
        stack = _escape_latex(", ".join(project.technologies))
        lines.extend(
            [
                r"  \item",
                f"  {heading}",
                r"  \textnormal{|}",
                rf"  \textit{{{stack}}}",
                "",
                r"  {\cvbody",
                r"  \begin{cvbullets}",
            ]
        )
        for bullet in project.bullets:
            lines.extend([rf"    \item {_escape_latex(bullet)}", ""])
        if lines[-1] == "":
            lines.pop()
        lines.extend([r"  \end{cvbullets}", r"  }", ""])
    if lines[-1] == "":
        lines.pop()
    lines.append(r"\end{cvblocks}")
    return "\n".join(lines) + "\n"


def replace_projects_section(base_resume: str, replacement: str, suffix: str) -> str:
    """Replace the Technical Projects section in a Markdown or LaTeX resume."""
    suffix = suffix.lower()
    if suffix == ".tex":
        heading = r"\section{Technical Projects}"
        next_heading = r"\section{"
    elif suffix in {".md", ".markdown"}:
        heading = "## Technical Projects"
        next_heading = "\n## "
    else:
        raise JobSpecificResumeError(
            "Base resume must be Markdown or LaTeX (.md, .markdown, .tex)"
        )

    start = base_resume.find(heading)
    if start < 0:
        raise JobSpecificResumeError(
            f"Base resume does not contain the {heading!r} section"
        )
    if suffix == ".tex":
        end = base_resume.find(next_heading, start + len(heading))
    else:
        end = base_resume.find(next_heading, start + len(heading))
        if end >= 0:
            end += 1
    if end < 0:
        end = base_resume.find(r"\end{document}", start) if suffix == ".tex" else -1
        if end < 0:
            end = len(base_resume)
    return base_resume[:start] + replacement.rstrip() + "\n\n" + base_resume[end:]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "job"


def write_tailoring_artifacts(
    *,
    result: TailoringResult,
    output_dir: Path,
    base_resume: Optional[Path] = None,
    filename_suffix: str = "",
) -> Dict[str, str]:
    """Write a resume copy plus an evidence manifest for one job."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stem = _slug(f"{result.company}-{result.job_title}-{result.job_id}")
    if filename_suffix:
        stem += "-" + _slug(filename_suffix)

    if base_resume:
        template_path = Path(base_resume).expanduser().resolve()
        if not template_path.exists():
            raise JobSpecificResumeError(f"Base resume not found: {template_path}")
        suffix = template_path.suffix.lower()
        replacement = (
            render_latex_projects(result)
            if suffix == ".tex"
            else render_markdown_projects(result)
        )
        resume_text = replace_projects_section(
            template_path.read_text(encoding="utf-8"), replacement, suffix
        )
    else:
        suffix = ".md"
        resume_text = render_markdown_projects(result)

    resume_path = destination / f"{stem}{suffix}"
    manifest_path = destination / f"{stem}.manifest.json"
    resume_path.write_text(resume_text, encoding="utf-8")

    match_map = {match.project.id: match for match in result.matches}
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job": {
            "id": result.job_id,
            "title": result.job_title,
            "company": result.company,
        },
        "resume_path": str(resume_path),
        "projects": [
            {
                "id": project.project_id,
                "name": project.name,
                "repository": project.repository,
                "match_score": match_map[project.project_id].score,
                "match_reason": match_map[project.project_id].reason(),
                "technologies": list(project.technologies),
                "bullets": [
                    {"text": text, "evidence_ids": list(evidence_ids)}
                    for text, evidence_ids in zip(project.bullets, project.evidence_ids)
                ],
            }
            for project in result.projects
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "resume_path": str(resume_path),
        "manifest_path": str(manifest_path),
    }


def catalog_to_prompt_preview(matches: Iterable[ProjectMatch]) -> List[Dict[str, Any]]:
    """Return compact serialisable match details for CLI previews and tests."""
    return [
        {
            "id": match.project.id,
            "name": match.project.name,
            "score": match.score,
            "reason": match.reason(),
            "needs_review": match.project.needs_review,
        }
        for match in matches
    ]


def compile_resume_pdf(
    source: Path, *, command: Sequence[str] = (), max_pages: int = 1
) -> Dict[str, Any]:
    """Compile a fresh LaTeX artifact and verify its PDF before upload."""
    from pypdf import PdfReader

    source = source.resolve()
    output = source.parent / "build"
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / f"{source.stem}.pdf"
    if pdf.exists():
        raise JobSpecificResumeError("Refusing to reuse an existing compiled PDF")
    if command:
        if isinstance(command, str):
            raise JobSpecificResumeError("compiler_command must be an argument list")
        # Native TeX engines on Windows can misparse non-ASCII absolute paths.
        # Compile from the artifact folder using English relative names.
        argv = [
            str(part).format(tex=source.name, output_dir=output.name)
            for part in command
        ]
    elif shutil.which("tectonic"):
        argv = ["tectonic", source.name, "--outdir", output.name, "--keep-logs"]
    elif shutil.which("xelatex"):
        argv = [
            "xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output.name}",
            source.name,
        ]
    else:
        raise JobSpecificResumeError(
            "No LaTeX compiler found; set precision_apply.compiler_command"
        )
    try:
        completed = subprocess.run(
            argv,
            cwd=source.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUTF8": "1"},
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise JobSpecificResumeError(f"LaTeX compilation failed: {exc}") from exc
    if completed.returncode != 0 or not pdf.is_file():
        raise JobSpecificResumeError(
            f"LaTeX compilation failed: {(completed.stderr or completed.stdout)[-1200:]}"
        )
    try:
        reader = PdfReader(pdf)
        pages = len(reader.pages)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise JobSpecificResumeError(f"Compiled PDF is unreadable: {exc}") from exc
    if not pages or not text.strip():
        raise JobSpecificResumeError("Compiled PDF has no extractable resume text")
    if pages > max_pages:
        raise JobSpecificResumeError(f"Resume exceeds {max_pages} page(s): {pages}")
    return {"resume_pdf_path": str(pdf), "resume_text": text, "pdf_pages": pages}


def load_tailoring_rules(config: Dict[str, Any]) -> str:
    """Read the configured policy afresh and reject missing or empty files."""
    value = config.get("precision_apply", {}).get("rules_file")
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    try:
        rules = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise JobSpecificResumeError(f"Cannot read tailoring rules: {path}") from exc
    if not rules:
        raise JobSpecificResumeError(f"Tailoring rules file is empty: {path}")
    return rules


def prepare_precision_resume(
    job: Dict[str, Any], config: Dict[str, Any], *, ai_service: Any = None
) -> Dict[str, Any]:
    """Prepare one evidence-backed PDF; failure never falls back to a generic CV."""
    settings = config.get("precision_apply", {})
    if not settings.get("enabled", False):
        return {}
    tailoring_rules = load_tailoring_rules(config)
    from ronin.config import get_ronin_home

    root = Path(__file__).resolve().parent.parent

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()

    if not settings.get("catalog") or not settings.get("base_resume"):
        raise JobSpecificResumeError("Precision apply requires catalog and base_resume")
    base = resolve(settings["base_resume"])
    if base.suffix.lower() != ".tex" or not base.is_file():
        raise JobSpecificResumeError(
            "Precision apply requires an existing LaTeX base resume"
        )
    # Validate the section before spending any AI allowance.
    replace_projects_section(base.read_text(encoding="utf-8"), "", ".tex")
    title = str(job.get("title") or "")
    company = str(job.get("company_name") or "")
    description = str(job.get("description") or "")
    if not title.strip() or not company.strip() or not description.strip():
        raise JobSpecificResumeError(
            "Precision apply requires title, company and full job description"
        )
    limit = int(settings.get("project_limit", 3))
    matches = select_projects(
        load_project_catalog(resolve(settings["catalog"])),
        job_title=title,
        job_description=description,
        limit=limit,
        min_score=float(settings.get("min_project_score", 5)),
    )
    if len(matches) != limit:
        raise JobSpecificResumeError(
            f"Need {limit} reviewed matching projects; found {len(matches)}"
        )
    if ai_service is None:
        from ronin.ai import CodexService

        ai_service = CodexService(default_model="gpt-5.6-terra", reasoning_effort="low")
    result = tailor_projects_with_ai(
        ai_service=ai_service,
        model=settings.get("model", "gpt-5.6-terra"),
        job_id=str(job.get("job_id") or "manual"),
        job_title=title,
        company=company,
        job_description=description,
        matches=matches,
        bullets_per_project=int(settings.get("bullets_per_project", 3)),
        max_words_per_bullet=int(settings.get("max_words_per_bullet", 36)),
        tailoring_rules=tailoring_rules,
    )
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = (
        resolve(settings["output_dir"])
        if settings.get("output_dir")
        else get_ronin_home() / "tailored_resumes"
    )
    artifacts = write_tailoring_artifacts(
        result=result,
        output_dir=destination / _slug(result.job_id) / run,
        base_resume=base,
        filename_suffix=run,
    )
    # Unique filenames distinguish this upload from all previous versions on SEEK.
    unique_source = Path(artifacts["resume_path"])
    compiled = compile_resume_pdf(
        unique_source,
        command=settings.get("compiler_command", ()),
        max_pages=int(settings.get("max_pages", 1)),
    )
    normalized = re.sub(r"\s+", "", compiled["resume_text"]).lower()
    for project in result.projects:
        if re.sub(r"\s+", "", project.name).lower() not in normalized:
            raise JobSpecificResumeError(
                f"Compiled PDF is missing selected project: {project.name}"
            )
    manifest_path = Path(artifacts["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        resume_path=str(unique_source),
        pdf_path=compiled["resume_pdf_path"],
        pdf_pages=compiled["pdf_pages"],
        tailoring_rules=tailoring_rules,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        **compiled,
        "selected_projects": json.dumps(
            [project.project_id for project in result.projects]
        ),
        "tailored_resume_path": compiled["resume_pdf_path"],
        "tailoring_manifest_path": str(manifest_path),
        "tailoring_generated_at": manifest["created_at"],
    }
