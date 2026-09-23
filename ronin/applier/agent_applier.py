"""Browser agent for external (link-out) job applications.

The external CLI existed before this module did.  This implementation keeps the
automation deliberately conservative: it fills fields from the saved Ronin
profile, uploads the configured resume, walks ordinary continue/review pages,
and stops for captchas, account creation, or unanswered required questions.
"""

from __future__ import annotations

import re
import shutil
import time
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from loguru import logger
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

from ronin.ai import AIService
from ronin.applier.browser import ChromeDriver
from ronin.applier.cover_letter import CoverLetterGenerator
from ronin.config import get_ronin_home, load_config
from ronin.profile import load_profile

STATUS_APPLIED = "APPLIED"
STATUS_DRY_RUN = "DRY_RUN"
STATUS_STALE = "STALE"
STATUS_NEEDS_HUMAN = "NEEDS_HUMAN"
STATUS_BLOCKED = "BLOCKED"
STATUS_STEP_BUDGET_EXHAUSTED = "STEP_BUDGET_EXHAUSTED"
STATUS_APP_ERROR = "APP_ERROR"


_SUBMIT_WORDS = (
    "submit application",
    "submit my application",
    "submit",
    "send application",
)
_NEXT_WORDS = (
    "autofill with resume",
    "apply manually",
    "apply now",
    "apply for this job",
    "continue",
    "next",
    "review",
    "save and continue",
    "proceed",
)
_SUCCESS_MARKERS = (
    "application submitted",
    "application has been submitted",
    "thank you for applying",
    "thanks for applying",
    "we have received your application",
    "application complete",
)
_STALE_MARKERS = (
    "job is no longer advertised",
    "position is no longer available",
    "job is no longer available",
    "applications are closed",
)
_BLOCKED_MARKERS = (
    "captcha",
    "verify you are human",
    "security challenge",
    "access denied",
    "cloudflare ray id",
)
_LOGIN_MARKERS = (
    "sign in to continue",
    "log in to continue",
)


def _normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_company_filename(company: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', " ", company).strip().rstrip(".")
    return re.sub(r"\s+", " ", value) or "Company"


class AgentApplier:
    """Fill and submit conventional external ATS forms in a real browser."""

    def __init__(self, dry_run: Optional[bool] = None) -> None:
        self.config = load_config()
        self.agent_config = self.config.get("agent_apply", {}) or {}
        configured = bool(self.agent_config.get("dry_run", True))
        self.dry_run = configured if dry_run is None else bool(dry_run)
        self.max_steps = max(1, int(self.agent_config.get("max_steps", 14) or 14))
        self.pause = max(
            0.2, float(self.agent_config.get("step_pause_seconds", 2.0) or 2.0)
        )
        self.profile = load_profile()
        self.browser = ChromeDriver()
        self.ai = AIService(
            default_model=str(self.agent_config.get("brain_model") or "gpt-5.6-terra")
        )
        self.cover_letters = CoverLetterGenerator(ai_service=self.ai)
        self._cover_letter = ""
        self._resume_text = ""
        self._resume_pdf: Optional[Path] = None

    @property
    def driver(self):
        return self.browser.driver

    def _profile_resume(self, resume_profile: str) -> Optional[object]:
        try:
            return self.profile.get_resume(resume_profile)
        except KeyError:
            try:
                return self.profile.get_resume("default")
            except KeyError:
                return self.profile.resumes[0] if self.profile.resumes else None

    def _resolve_resume_text(self, resume_profile: str) -> str:
        resume = self._profile_resume(resume_profile)
        if resume and resume.file:
            configured = Path(resume.file).expanduser()
            candidates = [configured, get_ronin_home() / "resumes" / configured]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.read_text(encoding="utf-8", errors="replace")
        return ""

    def _resolve_resume_pdf(self, resume_profile: str) -> Path:
        configured_dir = Path(
            str(self.agent_config.get("resume_pdf_dir") or "resume/pdf")
        ).expanduser()
        if not configured_dir.is_absolute():
            configured_dir = Path.cwd() / configured_dir
        file_map = self.agent_config.get("resume_pdf_map", {}) or {}
        mapped = str(file_map.get(resume_profile) or file_map.get("default") or "")
        candidates: list[Path] = []
        if mapped:
            mapped_path = Path(mapped).expanduser()
            candidates.append(
                mapped_path
                if mapped_path.is_absolute()
                else configured_dir / mapped_path
            )
        if configured_dir.is_dir():
            candidates.extend(sorted(configured_dir.glob("*.pdf"), reverse=True))
        resume = self._profile_resume(resume_profile)
        if resume and str(resume.file).lower().endswith(".pdf"):
            candidates.extend(
                [
                    Path(resume.file).expanduser(),
                    get_ronin_home() / "resumes" / resume.file,
                ]
            )
        candidates.extend(sorted((get_ronin_home() / "resumes").glob("*.pdf")))
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "No resume PDF found. Set agent_apply.resume_pdf_dir or "
            "agent_apply.resume_pdf_map in config.yaml."
        )

    def _page_text(self) -> str:
        try:
            return _normalise(self.driver.find_element(By.TAG_NAME, "body").text)
        except Exception:
            return ""

    def _page_state(self) -> Optional[str]:
        text = self._page_text().lower()
        url = str(self.driver.current_url or "").lower()
        if any(marker in text for marker in _SUCCESS_MARKERS):
            return STATUS_APPLIED
        if any(marker in text for marker in _STALE_MARKERS):
            return STATUS_STALE
        if any(marker in text for marker in _BLOCKED_MARKERS):
            return STATUS_BLOCKED
        if any(marker in text for marker in _LOGIN_MARKERS):
            return STATUS_NEEDS_HUMAN
        if any(
            part in url for part in ("/thank", "/confirmation", "application-success")
        ):
            return STATUS_APPLIED
        return None

    def _switch_to_new_window(self, previous: set[str]) -> None:
        deadline = time.time() + 12
        while time.time() < deadline:
            new_handles = set(self.driver.window_handles) - previous
            if new_handles:
                self.driver.switch_to.window(next(iter(new_handles)))
                return
            time.sleep(0.25)

    def _click_seek_apply(self) -> bool:
        selectors = (
            "[data-automation='job-detail-apply']",
            "a[href*='/apply']",
            "button[data-testid*='apply']",
        )
        for selector in selectors:
            for element in self.driver.find_elements(By.CSS_SELECTOR, selector):
                if not element.is_displayed() or not element.is_enabled():
                    continue
                previous = set(self.driver.window_handles)
                element.click()
                self._switch_to_new_window(previous)
                time.sleep(self.pause)
                return True
        return False

    def _open_application(self, apply_url: str, job_url: str) -> Optional[str]:
        target = _normalise(apply_url) or _normalise(job_url)
        if not target:
            return STATUS_APP_ERROR
        self.browser.navigate_to(target)
        time.sleep(self.pause)
        state = self._page_state()
        if state:
            return state
        host = (urlsplit(self.driver.current_url).hostname or "").lower()
        if "seek.com" in host:
            if not self._click_seek_apply() and job_url and target != job_url:
                self.browser.navigate_to(job_url)
                time.sleep(self.pause)
                if not self._click_seek_apply():
                    return STATUS_NEEDS_HUMAN
            deadline = time.time() + 20
            while time.time() < deadline:
                host = (urlsplit(self.driver.current_url).hostname or "").lower()
                if "seek.com" not in host:
                    break
                time.sleep(0.5)
        return self._page_state()

    def _field_label(self, element) -> str:
        try:
            return _normalise(
                self.driver.execute_script(
                    """
                    const e=arguments[0];
                    const labels=[];
                    if(e.labels) for(const l of e.labels) labels.push(l.innerText);
                    const p=e.closest('label,fieldset,[role="group"],.form-group,.field');
                    if(p) labels.push(p.innerText);
                    labels.push(e.getAttribute('aria-label'), e.placeholder, e.name, e.id);
                    return labels.filter(Boolean).join(' | ');
                    """,
                    element,
                )
            )
        except Exception:
            return " ".join(
                filter(
                    None,
                    [
                        element.get_attribute("aria-label"),
                        element.get_attribute("placeholder"),
                        element.get_attribute("name"),
                        element.get_attribute("id"),
                    ],
                )
            )

    @staticmethod
    def _is_required(element) -> bool:
        return bool(
            element.get_attribute("required")
            or element.get_attribute("aria-required") == "true"
        )

    def _known_answer(self, label: str, input_type: str) -> Optional[str]:
        text = label.lower()
        personal = self.profile.personal
        rights = self.profile.work_rights
        professional = self.profile.professional
        name_parts = personal.name.strip().split()
        first = name_parts[0] if name_parts else ""
        last = " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
        if input_type == "email" or "email" in text:
            return personal.email
        if input_type == "tel" or any(v in text for v in ("phone", "mobile")):
            return personal.phone
        if "first name" in text or "given name" in text:
            return first
        if any(v in text for v in ("last name", "surname", "family name")):
            return last
        if "full name" in text or text in {"name", "name *"}:
            return personal.name
        if any(v in text for v in ("city", "location", "suburb")):
            return personal.location
        if "notice" in text or "available to start" in text:
            return rights.notice_period
        if "visa" in text:
            return rights.visa_status
        if "citizenship" in text or "nationality" in text:
            return rights.citizenship
        if "security clearance" in text:
            return "None"
        if "years" in text and "experience" in text:
            return str(professional.years_experience)
        if "salary" in text or "remuneration" in text:
            return "Negotiable"
        if "linkedin" in text:
            return ""
        if "github" in text:
            match = re.search(r"https?://github\.com/[^\s)]+", self._resume_text)
            return match.group(0) if match else ""
        if "portfolio" in text or "website" in text:
            return ""
        return None

    def _choice_preference(self, label: str) -> list[str]:
        text = label.lower()
        rights = self.profile.work_rights
        if "security clearance" in text:
            return ["no", "none", "not held"]
        if "citizen" in text or "permanent resident" in text:
            return ["student visa", "temporary visa", "visa holder", "no"]
        if "work" in text and any(
            marker in text for marker in ("unrestricted", "full", "without restriction")
        ):
            return ["no", "student visa", "temporary visa"]
        if "work rights" in text or "right to work" in text or "visa" in text:
            return ["student visa", "temporary visa", "limited", "yes"]
        if "sponsor" in text and "future" in text:
            return ["yes"]
        if "sponsor" in text:
            return ["no"]
        if "driver" in text and "licen" in text:
            return ["yes" if rights.has_drivers_license else "no"]
        if "relocat" in text:
            return ["yes" if rights.willing_to_relocate else "no"]
        if "experience" in text:
            return ["less than 1 year", "under 1 year", "0-1", "0"]
        if "privacy" in text or "terms" in text or "consent" in text:
            return ["yes", "agree", "accept"]
        return []

    @staticmethod
    def _best_option(options: list[tuple[str, str]], preferences: list[str]):
        for preference in preferences:
            for value, label in options:
                if preference in label.lower():
                    return value
        return None

    def _fill_select(self, element, label: str) -> bool:
        select = Select(element)
        options = [
            (option.get_attribute("value") or "", _normalise(option.text))
            for option in select.options
            if option.is_enabled() and _normalise(option.text)
        ]
        value = self._best_option(options, self._choice_preference(label))
        if value is None:
            known = self._known_answer(label, "select")
            if known:
                value = self._best_option(options, [known.lower()])
        if value is None:
            return False
        select.select_by_value(value)
        return True

    def _fill_radio_group(self, element, label: str) -> bool:
        name = element.get_attribute("name")
        if not name:
            return False
        radios = self.driver.find_elements(
            By.CSS_SELECTOR, f'input[type="radio"][name="{name}"]'
        )
        options = []
        for radio in radios:
            radio_label = self._field_label(radio)
            options.append((radio.get_attribute("value") or "", radio_label, radio))
        preferences = self._choice_preference(label)
        for preference in preferences:
            for value, option_label, radio in options:
                if preference in f"{value} {option_label}".lower():
                    if not radio.is_selected():
                        self.driver.execute_script("arguments[0].click()", radio)
                    return True
        return False

    def _fill_text(self, element, label: str, input_type: str) -> bool:
        value = self._known_answer(label, input_type)
        lower = label.lower()
        if value is None and any(
            marker in lower
            for marker in ("cover letter", "why do you", "why are you", "motivation")
        ):
            value = self._cover_letter
        if value is None or value == "":
            return False
        element.clear()
        element.send_keys(value)
        return True

    def _fill_page(self) -> list[str]:
        unresolved: list[str] = []
        seen_radios: set[str] = set()
        for element in self.driver.find_elements(
            By.CSS_SELECTOR, "input, textarea, select"
        ):
            try:
                if not element.is_displayed() or not element.is_enabled():
                    continue
                tag = element.tag_name.lower()
                input_type = (element.get_attribute("type") or tag).lower()
                if input_type in {"hidden", "submit", "button", "reset", "image"}:
                    continue
                label = self._field_label(element)
                if input_type == "file":
                    if "cover" in label.lower():
                        if self._is_required(element):
                            unresolved.append(label or "cover-letter upload")
                    else:
                        element.send_keys(str(self._resume_pdf))
                    continue
                if input_type == "checkbox":
                    lower = label.lower()
                    if any(
                        v in lower for v in ("privacy", "terms", "consent", "agree")
                    ):
                        if not element.is_selected():
                            self.driver.execute_script("arguments[0].click()", element)
                        continue
                    if self._is_required(element) and not element.is_selected():
                        unresolved.append(label or "required checkbox")
                    continue
                if input_type == "radio":
                    name = element.get_attribute("name") or label
                    if name in seen_radios:
                        continue
                    seen_radios.add(name)
                    filled = self._fill_radio_group(element, label)
                elif tag == "select":
                    if element.get_attribute("value"):
                        continue
                    filled = self._fill_select(element, label)
                else:
                    if _normalise(element.get_attribute("value")):
                        continue
                    filled = self._fill_text(element, label, input_type)
                if not filled and self._is_required(element):
                    unresolved.append(
                        label or element.get_attribute("name") or input_type
                    )
            except StaleElementReferenceException:
                continue
            except Exception as exc:
                logger.debug(f"Could not fill external field {label!r}: {exc}")
                if self._is_required(element):
                    unresolved.append(label or input_type)
        return list(dict.fromkeys(unresolved))

    def _buttons(self, words: tuple[str, ...]):
        matches = []
        for element in self.driver.find_elements(
            By.CSS_SELECTOR, "button, input[type='submit'], a[role='button'], a"
        ):
            try:
                if not element.is_displayed() or not element.is_enabled():
                    continue
                text = _normalise(
                    element.text
                    or element.get_attribute("value")
                    or element.get_attribute("aria-label")
                ).lower()
                if any(text == word or text.startswith(word) for word in words):
                    matches.append(element)
            except StaleElementReferenceException:
                continue
        return matches

    def _archive_resume(self, company_name: str) -> None:
        if not self._resume_pdf:
            return
        repo = Path(__file__).resolve().parents[2]
        root = repo.parent
        destination = root / date.today().isoformat()
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            self._resume_pdf,
            destination / f"{_safe_company_filename(company_name)}.pdf",
        )

    def apply_to_job(
        self,
        *,
        job_id: str,
        job_description: str,
        score: int,
        key_tools: str,
        company_name: str,
        title: str,
        resume_profile: str,
        work_type: str,
        apply_url: Optional[str],
        job_url: Optional[str],
    ) -> str:
        del score
        try:
            self._resume_pdf = self._resolve_resume_pdf(resume_profile)
            self._resume_text = self._resolve_resume_text(resume_profile)
            letter = self.cover_letters.generate_cover_letter(
                job_description=job_description,
                title=title,
                company_name=company_name,
                key_tools=key_tools,
                resume_text=self._resume_text,
                work_type=work_type,
            )
            self._cover_letter = _normalise((letter or {}).get("response"))
            self.browser.initialize()
            opened = self._open_application(apply_url or "", job_url or "")
            if opened:
                return opened

            for _ in range(self.max_steps):
                state = self._page_state()
                if state:
                    if state == STATUS_APPLIED and not self.dry_run:
                        self._archive_resume(company_name)
                    return state

                unresolved = self._fill_page()
                submit = self._buttons(_SUBMIT_WORDS)
                if submit:
                    if unresolved:
                        logger.warning(
                            "External application requires human input: "
                            + "; ".join(unresolved[:5])
                        )
                        return STATUS_NEEDS_HUMAN
                    if self.dry_run:
                        return STATUS_DRY_RUN
                    submit[0].click()
                    time.sleep(self.pause * 2)
                    state = self._page_state()
                    if state == STATUS_APPLIED:
                        self._archive_resume(company_name)
                        return state
                    # Some ATSes show one final confirmation page.
                    continue

                next_buttons = self._buttons(_NEXT_WORDS)
                if unresolved:
                    logger.warning(
                        "External application requires human input: "
                        + "; ".join(unresolved[:5])
                    )
                    return STATUS_NEEDS_HUMAN
                if not next_buttons:
                    logger.warning(
                        f"No recognised continue/submit control for external job {job_id}"
                    )
                    return STATUS_NEEDS_HUMAN
                previous_url = self.driver.current_url
                next_buttons[0].click()
                time.sleep(self.pause)
                if self.driver.current_url == previous_url:
                    time.sleep(self.pause)
            return STATUS_STEP_BUDGET_EXHAUSTED
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return STATUS_NEEDS_HUMAN
        except Exception as exc:
            logger.exception(f"External application failed for {job_id}: {exc}")
            return STATUS_APP_ERROR

    def cleanup(self) -> None:
        try:
            self.ai.close()
        finally:
            self.browser.cleanup()
