"""Regression tests for the interactive setup wizard."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys

import yaml

from ronin.cli import setup as setup_cli
from ronin.cli.setup import STEP_ORDER, SetupWizard
from ronin.config import load_config
from ronin.profile import load_profile


def test_python_module_entrypoint_runs(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["RONIN_HOME"] = str(tmp_path / "module-home")
    result = subprocess.run(
        [sys.executable, "-m", "ronin", "--help"],
        cwd=Path(__file__).parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Ronin" in result.stdout


def test_personal_fields_autosave_and_restore(
    tmp_path: Path, monkeypatch
) -> None:
    ronin_home = tmp_path / "autosave-home"
    monkeypatch.setattr(setup_cli, "RONIN_HOME", ronin_home)

    async def enter_personal_data() -> None:
        app = SetupWizard(start_step="personal")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.screen.query_one("#name").value = "Reese Lee"
            app.screen.query_one("#email").value = "reese@example.com"
            await pilot.pause()

    asyncio.run(enter_personal_data())

    draft_path = ronin_home / "setup_draft.yaml"
    draft = yaml.safe_load(draft_path.read_text(encoding="utf-8"))
    assert draft["last_step"] == "personal"
    assert draft["wizard_data"]["personal"]["name"] == "Reese Lee"

    restored = SetupWizard()
    assert restored._step_index == STEP_ORDER.index("personal")
    assert restored.wizard_data["personal"]["email"] == "reese@example.com"


def test_personal_input_renders_typed_text() -> None:
    async def exercise_input() -> None:
        app = SetupWizard(start_step="personal")
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            field = app.screen.query_one("#name")
            field.focus()
            await pilot.press("r", "e", "e", "s", "e")
            await pilot.pause()

            rendered = "\n".join(
                field.render_line(line).text for line in range(field.size.height)
            )
            assert field.value == "reese"
            assert field.content_region.height >= 1
            assert "reese" in rendered

    asyncio.run(exercise_input())


def test_skill_input_renders_typed_text() -> None:
    async def exercise_skill_input() -> None:
        app = SetupWizard(start_step="professional")
        async with app.run_test(size=(140, 60)) as pilot:
            await pilot.pause()
            field = app.screen.query_one("#skill_core_skills")
            field.focus()
            await pilot.press("p", "y", "t", "h", "o", "n")
            await pilot.pause()

            rendered = "\n".join(
                field.render_line(line).text for line in range(field.size.height)
            )
            assert field.value == "python"
            assert field.content_region.height >= 1
            assert "python" in rendered

    asyncio.run(exercise_skill_input())


def test_next_button_mounts_every_setup_step() -> None:
    async def walk_wizard() -> None:
        app = SetupWizard(start_step="welcome")
        async with app.run_test(size=(140, 60)) as pilot:
            await pilot.pause()
            assert app._step_index == 0

            for expected_index in range(1, len(STEP_ORDER)):
                await pilot.click("#nav_next")
                await pilot.pause()
                assert app._step_index == expected_index

    asyncio.run(walk_wizard())


def test_complete_setup_writes_reloadable_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    ronin_home = tmp_path / "ronin-home"
    monkeypatch.setattr(setup_cli, "RONIN_HOME", ronin_home)

    async def complete_wizard() -> None:
        app = SetupWizard(start_step="welcome")
        async with app.run_test(size=(160, 70)) as pilot:
            await pilot.pause()
            await pilot.click("#nav_next")

            app.screen.query_one("#name").value = "Reese Lee"
            app.screen.query_one("#email").value = "reese@example.com"
            app.screen.query_one("#phone").value = "+61 400 000 000"
            app.screen.query_one("#location").value = "Sydney, NSW"
            await pilot.click("#nav_next")

            app.screen.query_one("#citizenship").value = "Chinese"
            app.screen.query_one("#visa_status").value = "Student visa (subclass 500)"
            app.screen.query_one("#notice_period").value = "Immediate"
            await pilot.click("#nav_next")

            app.screen.query_one("#title").value = "Software Engineering Student"
            app.screen.query_one("#years_experience").value = "0"
            app.screen.query_one("#skill_core_skills").value = (
                "Full-Stack Development, Automated Testing"
            )
            app.screen.query_one("#skill_tools_and_software").value = (
                "Python, TypeScript, Docker"
            )
            await pilot.click("#add_skill_cat")
            await pilot.pause()
            app.screen.query_one("#skill_category_5").value = "Python, TypeScript"
            await pilot.click("#nav_next")

            app.screen.query_one("#high_value_signals").value = (
                "Software engineering internship, Structured mentoring"
            )
            app.screen.query_one("#red_flags").value = (
                "Australian citizenship required, Unpaid internship"
            )
            app.screen.query_one("#wt_part_time").value = True
            app.screen.query_one("#arr_hybrid").value = True
            await pilot.click("#nav_next")

            app.screen.query_one("#res0_name").value = "software"
            app.screen.query_one("#res0_text").value = "Software engineering resume"
            app.screen.query_one("#res0_jt_part_time").value = True
            await pilot.click("#add_resume")
            await pilot.pause()
            app.screen.query_one("#res1_name").value = "data"
            app.screen.query_one("#res1_text").value = "Data engineering resume"
            await pilot.click("#nav_next")

            app.screen.query_one("#anti_slop_rules").value = (
                "No invented metrics, Use Australian English"
            )
            await pilot.click("#nav_next")

            app.screen.query_one("#keywords").value = (
                "software engineering intern, graduate developer"
            )
            app.screen.query_one("#search_location").value = "Sydney"
            await pilot.click("#nav_next")

            app.screen.query_one("#seek_id_0").value = "seek-software-id"
            app.screen.query_one("#seek_id_1").value = "seek-data-id"
            await pilot.click("#nav_next")

            await pilot.click("#test_connection")
            await pilot.click("#nav_next")
            await pilot.click("#nav_next")
            await pilot.click("#nav_next")
            assert app._step_index == len(STEP_ORDER) - 1
            await pilot.click("#nav_next")

    asyncio.run(complete_wizard())

    profile_path = ronin_home / "profile.yaml"
    config_path = ronin_home / "config.yaml"
    env_path = ronin_home / ".env"
    assert profile_path.is_file()
    assert config_path.is_file()
    assert env_path.is_file()
    assert (ronin_home / "resumes" / "software.txt").read_text(
        encoding="utf-8"
    ) == "Software engineering resume"
    assert (ronin_home / "resumes" / "data.txt").read_text(
        encoding="utf-8"
    ) == "Data engineering resume"

    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert profile["personal"]["name"] == "Reese Lee"
    assert profile["professional"]["skills"]["core_skills"] == [
        "Full-Stack Development",
        "Automated Testing",
    ]
    assert profile["professional"]["skills"]["category_5"] == [
        "Python",
        "TypeScript",
    ]
    assert [resume["name"] for resume in profile["resumes"]] == [
        "software",
        "data",
    ]
    assert profile["resumes"][0]["seek_resume_id"] == "seek-software-id"
    assert config["search"]["keywords"] == [
        "software engineering intern",
        "graduate developer",
    ]
    assert config["search"]["location"] == "Sydney"

    validated_profile = load_profile(profile_path)
    monkeypatch.setenv("RONIN_HOME", str(ronin_home))
    reloaded_config = load_config()
    assert validated_profile.personal.name == "Reese Lee"
    assert [resume.name for resume in validated_profile.resumes] == [
        "software",
        "data",
    ]
    assert reloaded_config["search"]["location"] == "Sydney"
