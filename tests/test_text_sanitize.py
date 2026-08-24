import unittest

from src.mapper import Company
from src.text_sanitize import (
    build_handle_to_company_name,
    collect_numeric_tickers,
    sanitize_message_text,
)

_COMPANIES = [
    Company(
        company_id="tesla",
        company_name="Tesla",
        exchange="NASDAQ",
        ticker="TSLA",
        aliases=("Tesla", "特斯拉"),
        brands=(),
        negative_contexts=(),
    ),
    Company(
        company_id="kuaishou",
        company_name="Kuaishou",
        exchange="HKEX",
        ticker="1024",
        aliases=("Kuaishou",),
        brands=(),
        negative_contexts=(),
    ),
]


class TextSanitizeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.handle_map = build_handle_to_company_name(_COMPANIES)
        self.numeric_tickers = collect_numeric_tickers(_COMPANIES)

    def test_url_is_redacted(self) -> None:
        text = "check this out https://example.com/some/path?x=1"
        self.assertNotIn("example.com", sanitize_message_text(text))
        self.assertIn("[URL]", sanitize_message_text(text))

    def test_email_is_redacted(self) -> None:
        text = "contact me at someone@example.com for details"
        result = sanitize_message_text(text)
        self.assertNotIn("someone@example.com", result)
        self.assertIn("[EMAIL]", result)

    def test_official_handle_maps_to_company_name(self) -> None:
        text = "big news from @Tesla today"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertIn("Tesla", result)
        self.assertNotIn("[HANDLE]", result)

    def test_fan_account_handle_is_redacted_not_mapped(self) -> None:
        text = "@Tesla_Teslaway posted something unrelated"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertNotIn("Tesla_Teslaway", result)
        self.assertIn("[HANDLE]", result)

    def test_unknown_handle_is_redacted(self) -> None:
        text = "shoutout to @some_random_user_42"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertNotIn("some_random_user_42", result)
        self.assertIn("[HANDLE]", result)

    def test_long_id_is_redacted(self) -> None:
        text = "block 49145375 just landed"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertNotIn("49145375", result)
        self.assertIn("[ID]", result)

    def test_registered_numeric_ticker_is_preserved(self) -> None:
        text = "HKEX 1024 rallied today"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertIn("1024", result)
        self.assertNotIn("[ID]", result)

    def test_cashtag_is_untouched(self) -> None:
        text = "$NVDA and $AAPL both up today, strike $200"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertIn("$NVDA", result)
        self.assertIn("$AAPL", result)
        self.assertIn("$200", result)

    def test_bare_at_symbol_used_as_preposition_is_untouched(self) -> None:
        text = "live @ 5pm, see you there"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertIn("live @ 5pm", result)

    def test_short_platform_handle_is_untouched(self) -> None:
        text = "posted this on @X earlier"
        result = sanitize_message_text(text, self.handle_map, self.numeric_tickers)
        self.assertIn("@X", result)


if __name__ == "__main__":
    unittest.main()
