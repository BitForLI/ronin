from ronin.applier.browser import ChromeDriver


class FakeDriver:
    def __init__(self, url: str, indicator: bool = True):
        self.current_url = url
        self.indicator = indicator

    def execute_script(self, _script: str) -> bool:
        return self.indicator


def test_google_login_page_is_never_authenticated() -> None:
    browser = ChromeDriver()
    browser.driver = FakeDriver("https://accounts.google.com/signin", True)
    assert browser._check_logged_in_indicators() is False


def test_seek_sign_in_page_is_never_authenticated() -> None:
    browser = ChromeDriver()
    browser.driver = FakeDriver("https://www.seek.com.au/sign-in", True)
    assert browser._check_logged_in_indicators() is False


def test_authenticated_seek_indicator_is_accepted() -> None:
    browser = ChromeDriver()
    browser.driver = FakeDriver("https://www.seek.com.au/profile/me", True)
    assert browser._check_logged_in_indicators() is True
