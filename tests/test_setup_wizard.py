"""Regression tests for the interactive setup wizard."""

import asyncio

from ronin.cli.setup import SetupWizard


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
