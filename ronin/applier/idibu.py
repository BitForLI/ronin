"""Prepare and review Idibu/Applr applications without importing a default CV."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import Select, WebDriverWait

from ronin.profile import Profile


class IdibuApplication:
    """Adapter for the inspected Applr form, with an explicit submit boundary."""

    def __init__(self, driver: WebDriver, profile: Profile):
        self.driver = driver
        self.profile = profile
        self.review: Optional[Dict[str, Any]] = None
        self._privacy_element = None
        self._expected_fields: List[tuple] = []

    @staticmethod
    def supports(driver: WebDriver) -> bool:
        """Match the actual ATS form, not merely a button advertising SEEK."""
        if (urlsplit(driver.current_url).hostname or "").lower() != "applr.io":
            return False
        forms = driver.find_elements(By.CSS_SELECTOR, "form#form")
        if len(forms) != 1:
            return False
        action = urlsplit(forms[0].get_attribute("action") or "")
        if action.hostname != "applr.io" or not re.fullmatch(
            r"/jobs/\d+/submit_application", action.path
        ):
            return False
        return all(
            len(forms[0].find_elements(By.CSS_SELECTOR, selector)) == 1
            for selector in (
                "input#resume[type='file']",
                "input#first_name",
                "input#last_name",
                "input#email",
            )
        )

    def _profile_values(self) -> Dict[str, str]:
        """Use saved facts; an eligible future visa is never a current visa."""
        personal = self.profile.personal
        name = personal.name.split()
        if len(name) != 2 or not personal.email or not personal.phone:
            raise ValueError(
                "Confirm first/last name, email and phone in the saved profile"
            )
        location = personal.location.lower()
        regions = {
            "new south wales": ("nsw", "sydney"),
            "victoria": ("vic", "melbourne"),
            "queensland": ("qld", "brisbane"),
            "western australia": ("wa", "perth"),
            "south australia": ("sa", "adelaide"),
            "tasmania": ("tas", "hobart"),
            "australian capital territory": ("act", "canberra"),
            "northern territory": ("nt", "darwin"),
        }
        region = ""
        for full, aliases in regions.items():
            if any(
                re.search(r"\b" + re.escape(token) + r"\b", location)
                for token in (full, *aliases)
            ):
                region = full.title()
                break
        if not region:
            raise ValueError(
                "Confirm the residential Australian state in the saved profile"
            )
        rights = self.profile.work_rights
        current_visa = re.split(
            r";|eligible to|may apply|can apply|will apply",
            rights.visa_status,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        subclass = re.search(r"\b(?:subclass\s*)?(\d{3})\b", current_visa)
        if subclass:
            work_rights = "subclass " + subclass.group(1)
        elif "student" in current_visa.lower():
            work_rights = "subclass 500"
        elif rights.citizenship.lower().strip() in {
            "australian",
            "australian citizen",
            "australia",
        }:
            work_rights = "australian citizen"
        elif "permanent resident" in current_visa.lower():
            work_rights = "australian permanent resident"
        else:
            raise ValueError("Current work-right status is not unambiguously recorded")
        clearances = rights.security_clearances
        if len(clearances) > 1:
            raise ValueError("Confirm which current security clearance should be used")
        return {
            "first_name": name[0],
            "last_name": name[1],
            "email": personal.email,
            "phone": personal.phone,
            "region": region,
            "country": "Australia",
            "work_rights": work_rights,
            "clearance": clearances[0] if clearances else "None",
        }

    def _fields(self) -> list:
        """Read dynamic field labels in one DOM call, excluding hidden tokens."""
        return self.driver.execute_script(
            """
            return Array.from(document.querySelectorAll('#form input, #form select, #form textarea'))
                .filter(e => !['hidden','submit','button'].includes(e.type) && !e.disabled)
                .map(e => {
                    let p=e.parentElement, context='';
                    for(let i=0;p && i<4;i++,p=p.parentElement){
                        context=p.innerText || '';
                        if(/privacy policy/i.test(context)) break;
                    }
                    return {element:e, id:e.id, type:e.type || e.tagName.toLowerCase(),
                        required:e.required, label:Array.from(e.labels || [])
                            .map(l=>l.innerText.trim()).join(' '), context:context.slice(0,700),
                        accept:e.accept || '', options:e.tagName==='SELECT'
                            ? Array.from(e.options).map(o=>({text:o.text.trim(),value:o.value})) : []};
                });
        """
        )

    @staticmethod
    def _field_key(field: dict) -> str:
        if field["id"] in {"first_name", "last_name", "email"}:
            return field["id"]
        label = " ".join(field["label"].lower().replace("*", "").split())
        return {
            "mobile phone": "phone",
            "phone": "phone",
            "state/region": "region",
            "country": "country",
            "work right status": "work_rights",
            "security clearance": "clearance",
        }.get(label, "")

    @staticmethod
    def _option(field: dict, value: str) -> dict:
        wanted = value.lower().strip()
        choices = [o for o in field["options"] if o["value"]]
        matches = [o for o in choices if o["text"].lower().strip() == wanted]
        if not matches and wanted.startswith("subclass "):
            matches = [
                o
                for o in choices
                if re.match(re.escape(wanted) + r"\b", o["text"].lower())
            ]
        if len(matches) != 1:
            raise ValueError(f"No unique option for {field['label']}: {value}")
        return matches[0]

    def prepare(
        self,
        pdf_path: str,
        cover_letter: str,
        output_dir: Path,
        company: str,
        title: str,
    ) -> Dict[str, Any]:
        """Fill known fields and attach the exact PDF; leave privacy/submit untouched."""
        if not self.supports(self.driver):
            raise ValueError("This is not a supported Idibu application form")
        pdf = Path(pdf_path).resolve()
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError("A real application PDF is required")
        with pdf.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("Application document is not a PDF")
        if not cover_letter.strip():
            raise ValueError("Company-specific cover letter generation failed")
        values = self._profile_values()
        fields = self._fields()
        files = [f for f in fields if f["type"] == "file"]
        resume = [f for f in files if f["id"] == "resume"]
        letters = [f for f in files if f["id"] != "resume" and not f["required"]]
        if len(resume) != 1 or len(letters) != 1 or len(files) != 2:
            raise ValueError("Cannot uniquely identify CV and cover-letter uploads")
        if letters[0]["accept"] and ".txt" not in letters[0]["accept"].lower():
            raise ValueError("This cover-letter input does not accept plain text")
        plan = []
        privacy = []
        required_radios = [
            field for field in fields if field["type"] == "radio" and field["required"]
        ]
        single_privacy_choice = (
            len(required_radios) == 1
            and required_radios[0]["label"].lower().strip() == "yes"
            and "do you agree to our privacy policy?"
            in self.driver.find_element(By.TAG_NAME, "body").text.lower()
        )
        for field in fields:
            if field["type"] == "file":
                continue
            if field["type"] == "radio" and (
                "privacy policy" in field["context"].lower()
                or (single_privacy_choice and field is required_radios[0])
            ):
                if field["label"].lower().strip() == "yes":
                    privacy.append(field["element"])
                continue
            key = self._field_key(field)
            if key:
                value = values[key]
                option = self._option(field, value) if field["options"] else None
                plan.append(
                    (
                        field,
                        option["value"] if option else value,
                        option["text"] if option else value,
                    )
                )
            elif field["required"]:
                raise ValueError(
                    f"Unmapped required field: {field['label'] or field['id']}"
                )
        if len(privacy) != 1:
            raise ValueError("Cannot identify the privacy-consent choice")
        # Complete the plan before any applicant data or document is uploaded.
        output_dir.mkdir(parents=True, exist_ok=True)
        letter_path = output_dir / "cover-letter.txt"
        if (
            letter_path.exists()
            and letter_path.read_text(encoding="utf-8") != cover_letter
        ):
            raise ValueError(
                "A different cover letter already exists; choose a new output directory"
            )
        if not letter_path.exists():
            letter_path.write_text(cover_letter, encoding="utf-8")
        answers = {}
        self._expected_fields = []
        for field, value, display in plan:
            element = field["element"]
            if field["options"]:
                Select(element).select_by_value(value)
            else:
                element.clear()
                element.send_keys(value)
            self._expected_fields.append((element, value))
            answers[field["label"] or field["id"]] = display
        resume[0]["element"].send_keys(str(pdf))
        letters[0]["element"].send_keys(str(letter_path.resolve()))
        expected_uploads = [
            (resume[0]["element"], pdf.name),
            (letters[0]["element"], letter_path.name),
        ]
        WebDriverWait(self.driver, 30).until(
            lambda d: all(
                d.execute_script(
                    "return arguments[0].files.length===1 && arguments[0].files[0].name===arguments[1]",
                    element,
                    name,
                )
                for element, name in expected_uploads
            )
        )
        WebDriverWait(self.driver, 30).until(
            lambda d: all(
                name in d.find_element(By.TAG_NAME, "body").text
                for _, name in expected_uploads
            )
        )
        # CV parsing can pre-fill contact fields asynchronously. Saved facts
        # win over parsed guesses, especially work rights and residential country.
        for field, value, _ in plan:
            element = field["element"]
            if str(element.get_attribute("value")) != value:
                if field["options"]:
                    Select(element).select_by_value(value)
                else:
                    element.clear()
                    element.send_keys(value)
        self._privacy_element = privacy[0]
        self.review = {
            "company": company,
            "title": title,
            "submitted": False,
            "resume_pdf_path": str(pdf),
            "cover_letter_path": str(letter_path.resolve()),
            "answers": answers,
            "seek_profile_access_granted": False,
            "privacy_consent_granted": self._privacy_element.is_selected(),
        }
        (output_dir / "review.json").write_text(
            json.dumps(self.review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.driver.save_screenshot(str(output_dir / "review.png"))
        self._expected_uploads = expected_uploads
        return self.review

    def submit(self, output_dir: Path) -> bool:
        """Submit only after caller approval, checking that reviewed data is unchanged."""
        if self.review is None or not self.supports(self.driver):
            raise ValueError("No reviewed Idibu application is ready")
        if any(
            str(e.get_attribute("value")) != value for e, value in self._expected_fields
        ):
            raise ValueError("Application fields changed after review")
        if not all(
            self.driver.execute_script(
                "return arguments[0].files.length===1 && arguments[0].files[0].name===arguments[1]",
                element,
                name,
            )
            for element, name in self._expected_uploads
        ):
            raise ValueError("Application attachments changed after review")
        if not self._privacy_element.is_selected():
            self._privacy_element.click()
        if not self.driver.execute_script(
            "return document.querySelector('#form').checkValidity()"
        ):
            raise ValueError("The employer form still has invalid required fields")
        buttons = self.driver.find_elements(
            By.CSS_SELECTOR, "#form #form_submit[type='submit']"
        )
        if (
            len(buttons) != 1
            or not buttons[0].is_displayed()
            or not buttons[0].is_enabled()
        ):
            raise ValueError("Cannot identify the employer's final submit control")
        buttons[0].click()

        def confirmed(d: WebDriver) -> bool:
            if d.find_elements(By.CSS_SELECTOR, "form#form"):
                return False
            body = d.find_element(By.TAG_NAME, "body").text.lower()
            return bool(
                re.search(
                    r"(?:your )?application (?:has been |was )?(?:successfully )?(?:submitted|received|sent)|"
                    r"thank you for (?:your application|applying)",
                    body,
                )
            )

        WebDriverWait(self.driver, 30).until(confirmed)
        (output_dir / "confirmation.txt").write_text(
            self.driver.find_element(By.TAG_NAME, "body").text, encoding="utf-8"
        )
        self.driver.save_screenshot(str(output_dir / "confirmation.png"))
        self.review["submitted"] = True
        self.review["privacy_consent_granted"] = True
        self.review["submitted_at"] = datetime.now().isoformat()
        (output_dir / "review.json").write_text(
            json.dumps(self.review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
