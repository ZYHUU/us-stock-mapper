import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from src.database import (
    annotation_summary,
    database_counts,
    list_annotation_items,
    save_database_annotation,
    save_message,
)


class DatabaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "test.db"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def sample_record(self) -> dict[str, object]:
        return {
            "source_id": "123",
            "source_type": "twitter",
            "source_url": "https://example.com/123",
            "published_at": "2026-08-21 00:00:00",
            "text": "英伟达发布新芯片",
            "received_at": "2026-08-21T00:00:00+00:00",
            "status": "matched",
            "predicted_codes": ["NASDAQ:NVDA"],
            "companies": [
                {
                    "company_id": "nvidia",
                    "company_name": "英伟达",
                    "canonical_code": "NASDAQ:NVDA",
                    "mention": "英伟达",
                    "match_type": "alias",
                    "confidence": 0.98,
                }
            ],
            "upstream_candidates": [
                {
                    "stocker_id": "6176",
                    "stocker_code": "NOW",
                    "stocker_name": "现在服务公司",
                    "aliases": ["now"],
                    "type": "1",
                }
            ],
            "raw_event": {"id_str": "123", "content": "英伟达发布新芯片"},
        }

    def test_initialize_creates_security_master_tables(self) -> None:
        from src.database import connect, initialize

        initialize(self.database_path)
        with closing(connect(self.database_path)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }

        self.assertIn("securities", tables)
        self.assertIn("platform_instruments", tables)

    def test_message_is_persistently_deduplicated(self) -> None:
        first_id, first_inserted = save_message(
            self.sample_record(), "test-rules", self.database_path
        )
        second_id, second_inserted = save_message(
            self.sample_record(), "test-rules", self.database_path
        )

        self.assertTrue(first_inserted)
        self.assertFalse(second_inserted)
        self.assertEqual(first_id, second_id)
        self.assertEqual(
            database_counts(self.database_path),
            {
                "messages": 1,
                "predictions": 1,
                "upstream_candidates": 1,
                "annotations": 0,
            },
        )

    def test_annotation_uses_latest_append_only_value(self) -> None:
        save_message(self.sample_record(), "test-rules", self.database_path)
        save_database_annotation(
            "id:123",
            ["NASDAQ:NVDA"],
            ["NASDAQ:NVDA"],
            "human",
            "high",
            self.database_path,
        )
        save_database_annotation(
            "id:123",
            [],
            ["NASDAQ:NVDA"],
            "human",
            "high",
            self.database_path,
        )

        items = list_annotation_items(
            self.database_path,
            unlabeled_only=False,
        )
        self.assertEqual(items[0]["annotation"]["correct_codes"], [])
        self.assertEqual(items[0]["annotation"]["decision"], "no_tracked_company")
        self.assertEqual(
            annotation_summary(self.database_path),
            {"total": 1, "labeled": 1, "remaining": 0},
        )


if __name__ == "__main__":
    unittest.main()
