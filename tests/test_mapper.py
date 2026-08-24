import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.annotation_store import load_annotations, record_key, save_annotation
from src.mapper import default_mapper
from src.message_parser import (
    detect_source_type,
    extract_text,
    extract_upstream_candidates,
    identify_event,
)
from src.ws_client import _event_key, _events_from_payload


class USStockMapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapper = default_mapper()

    def codes_for(self, message: str) -> list[str]:
        return [item.canonical_code for item in self.mapper.identify(message)]

    def test_company_alias(self) -> None:
        self.assertEqual(self.codes_for("英伟达发布了新芯片"), ["NASDAQ:NVDA"])

    def test_brand(self) -> None:
        self.assertEqual(self.codes_for("iPhone销量创下新高"), ["NASDAQ:AAPL"])

    def test_ticker(self) -> None:
        self.assertEqual(self.codes_for("NVDA surged after earnings"), ["NASDAQ:NVDA"])

    def test_ticker_is_case_sensitive(self) -> None:
        self.assertEqual(
            self.codes_for("python -m unittest discover -s tests -v"),
            [],
        )

    def test_dollar_prefixed_ticker(self) -> None:
        self.assertEqual(self.codes_for("$NVDA surged after earnings"), ["NASDAQ:NVDA"])

    def test_single_letter_ticker_requires_dollar_prefix(self) -> None:
        self.assertEqual(self.codes_for("Vitamin C and V-shaped recovery"), [])
        self.assertEqual(self.codes_for("$C rose after earnings"), ["NYSE:C"])

    def test_multiple_companies(self) -> None:
        self.assertCountEqual(
            self.codes_for("微软与甲骨文扩大云计算合作"),
            ["NASDAQ:MSFT", "NYSE:ORCL"],
        )

    def test_sqlite_security_master_companies_are_enabled(self) -> None:
        self.assertEqual(
            self.codes_for("ServiceNow 发布季度财报"),
            ["NYSE:NOW"],
        )
        self.assertEqual(
            self.codes_for("泡泡玛特与长鑫科技发布新消息"),
            ["SSE:688825", "HKEX:9992"],
        )
        self.assertEqual(
            self.codes_for("Micron Technology expands production"),
            ["NASDAQ:MU"],
        )

    def test_non_company_tradfi_instruments_are_not_enabled(self) -> None:
        self.assertEqual(self.codes_for("QQQ and XAU moved higher"), [])
        self.assertEqual(self.codes_for("We need to act now"), [])

    def test_unknown_message(self) -> None:
        self.assertEqual(self.codes_for("今天市场整体表现平淡"), [])

    def test_ambiguous_word_with_negative_context(self) -> None:
        self.assertEqual(self.codes_for("红富士苹果批发价格下降"), [])
        self.assertEqual(self.codes_for("This custard apple tastes sweet"), [])

    def test_ambiguous_word_with_company_context(self) -> None:
        self.assertEqual(self.codes_for("苹果发布新款 iPhone"), ["NASDAQ:AAPL"])

    def test_english_keyword_requires_word_boundary(self) -> None:
        self.assertEqual(self.codes_for("The example is incomplete"), [])

    def test_extract_ws_text_and_remove_duplicate_title(self) -> None:
        event = {
            "content": "英伟达发布新一代芯片",
            "content_cn": "",
            "title": "英伟达发布新一代芯片",
            "new_full_text": "请核实新闻真实性",
            "quo_text": "",
        }
        self.assertEqual(
            extract_text(event),
            "英伟达发布新一代芯片\n请核实新闻真实性",
        )

    def test_ws_event_without_company(self) -> None:
        event = {
            "id_str": "2090441890583715898",
            "item_url": "https://twitter.com/grok/status/2090441890583715898",
            "content": "部分属实但夸大。荷兰部分市政当局可通过特殊补助个案报销。",
            "title": "部分属实但夸大。荷兰部分市政当局可通过特殊补助个案报销。",
            "new_full_text": "核实真实性，引用新闻来源",
        }
        result = identify_event(event, self.mapper)
        self.assertEqual(result["source_id"], "2090441890583715898")
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["companies"], [])

    def test_upstream_stock_candidate_is_not_a_final_match(self) -> None:
        event = {
            "content": "We need to act now before it is too late.",
            "stoks": [
                {
                    "aliases": ["now"],
                    "stocker_code": "NOW",
                    "stocker_id": 6176,
                    "stocker_name": "现在服务公司",
                    "type": "1",
                }
            ],
        }
        result = identify_event(event, self.mapper)
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["companies"], [])
        self.assertEqual(result["upstream_candidates"][0]["stocker_code"], "NOW")

    def test_upstream_candidates_accept_json_string_and_stocks_alias(self) -> None:
        value = '[{"stocker_code":"nvda","stocker_name":"英伟达"}]'
        candidates = extract_upstream_candidates({"stocks": value})
        self.assertEqual(candidates[0]["stocker_code"], "NVDA")

    def test_news_website_event_is_normalized(self) -> None:
        event = {
            "id": 4766112,
            "source_type": "news",
            "content_cn": "Injective 推出了与 SpaceX 私营公司股份相关的市场。",
            "content_url": "https://www.odaily.news/zh-CN/newsflash/511584",
            "create_time": "2026-08-21 14:43:21",
            "stocks": [
                {
                    "stocker_id": 7580,
                    "stocker_code": "SPCX",
                    "stocker_name": "SpaceX",
                    "type": "1",
                    "aliases": ["spacex"],
                }
            ],
        }
        result = identify_event(event, self.mapper)
        self.assertEqual(detect_source_type(event), "news")
        self.assertEqual(result["source_id"], "4766112")
        self.assertEqual(result["source_type"], "news")
        self.assertEqual(result["source_url"], event["content_url"])
        self.assertEqual(result["published_at"], "2026-08-21 14:43:21")
        self.assertEqual(result["status"], "matched")
        self.assertEqual(
            [company["canonical_code"] for company in result["companies"]],
            ["NASDAQ:SPCX"],
        )
        self.assertEqual(result["upstream_candidates"][0]["stocker_code"], "SPCX")

    def test_ws_data_envelope(self) -> None:
        event = {"id_str": "123", "content": "特斯拉发布新车"}
        self.assertEqual(list(_events_from_payload({"data": event})), [event])

    def test_ws_envelope_keeps_outer_stock_candidates(self) -> None:
        payload = {
            "data": {"id_str": "123", "content": "ServiceNow 发布财报"},
            "stoks": [{"stocker_code": "NOW"}],
        }
        event = list(_events_from_payload(payload))[0]
        self.assertEqual(event["stoks"][0]["stocker_code"], "NOW")

    def test_ws_event_key_prefers_source_id(self) -> None:
        self.assertEqual(_event_key({"id_str": "123", "content": "消息"}), "id:123")

    def test_companies_discovered_from_ws_samples(self) -> None:
        self.assertEqual(
            self.codes_for("万豪酒店集团旗下 Marriott Explore 员工价格"),
            ["NASDAQ:MAR"],
        )
        self.assertEqual(
            self.codes_for("花旗集团董事长兼 CEO Jane Fraser"),
            ["NYSE:C"],
        )

    def test_generic_product_words_do_not_match(self) -> None:
        self.assertEqual(self.codes_for("The Office was a popular TV show"), [])
        self.assertEqual(self.codes_for("This is the core design principle"), [])
        self.assertEqual(self.codes_for("Summarizing discussion threads"), [])

    def test_urls_do_not_trigger_company_matches(self) -> None:
        self.assertEqual(
            self.codes_for("https://example.com/story?utm_medium=iOS_share"),
            [],
        )
        self.assertEqual(self.codes_for("https://github.com/example/repo"), [])

    def test_apple_products(self) -> None:
        self.assertEqual(
            self.codes_for("泄露的 AirPods 摄像头可以配合 Siri 使用"),
            ["NASDAQ:AAPL"],
        )

    def test_annotation_is_append_only_and_latest_value_wins(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.jsonl"
            key = record_key({"source_id": "123", "text": "消息"})
            save_annotation(path, key, ["NASDAQ:NVDA"], ["NASDAQ:NVDA"])
            save_annotation(
                path,
                key,
                [],
                ["NASDAQ:NVDA"],
                annotator="codex_reviewed",
                confidence="medium",
            )
            annotations = load_annotations(path)

        self.assertEqual(annotations[key]["decision"], "no_tracked_company")
        self.assertEqual(annotations[key]["correct_codes"], [])
        self.assertEqual(annotations[key]["annotator"], "codex_reviewed")
        self.assertEqual(annotations[key]["confidence"], "medium")


if __name__ == "__main__":
    unittest.main()
