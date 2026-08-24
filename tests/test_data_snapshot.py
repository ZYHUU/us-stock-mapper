import hashlib
import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from src.create_data_snapshot import create_snapshot
from src.database import save_database_annotation, save_message


class DataSnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / "live.db"
        self.companies_path = self.root / "companies.csv"
        self.companies_path.write_text(
            "company_id,company_name,exchange,ticker,aliases,brands,negative_contexts\n"
            "nvidia,英伟达,NASDAQ,NVDA,英伟达|NVIDIA,,\n",
            encoding="utf-8",
            newline="\n",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def sample_record(source_id: str) -> dict[str, object]:
        return {
            "source_id": source_id,
            "source_type": "twitter",
            "source_url": f"https://example.com/{source_id}",
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
            "upstream_candidates": [],
            "raw_event": {"id_str": source_id},
        }

    def test_snapshot_stays_fixed_when_live_database_grows(self) -> None:
        save_message(self.sample_record("1"), "test-rules", self.database_path)
        save_database_annotation(
            "id:1",
            ["NASDAQ:NVDA"],
            ["NASDAQ:NVDA"],
            "human",
            "high",
            self.database_path,
        )

        manifest = create_snapshot(
            self.database_path,
            self.companies_path,
            self.root / "snapshots",
            "training-v1",
        )
        snapshot_directory = Path(manifest["snapshot_directory"])
        snapshot_database = snapshot_directory / "stock_mapper.db"

        save_message(self.sample_record("2"), "test-rules", self.database_path)

        with closing(sqlite3.connect(snapshot_database)) as connection:
            snapshot_messages = connection.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]
        with closing(sqlite3.connect(self.database_path)) as connection:
            live_messages = connection.execute(
                "SELECT COUNT(*) FROM messages"
            ).fetchone()[0]

        self.assertEqual(snapshot_messages, 1)
        self.assertEqual(live_messages, 2)
        self.assertEqual(manifest["training_data"]["labeled_messages"], 1)
        self.assertEqual(manifest["training_data"]["positive_messages"], 1)
        self.assertEqual(manifest["integrity_check"], "ok")
        self.assertTrue((snapshot_directory / "companies.csv").is_file())

        stored_manifest = json.loads(
            (snapshot_directory / "manifest.json").read_text(encoding="utf-8")
        )
        digest = hashlib.sha256(snapshot_database.read_bytes()).hexdigest()
        self.assertEqual(stored_manifest["files"]["database"]["sha256"], digest)

    def test_existing_snapshot_is_not_overwritten(self) -> None:
        save_message(self.sample_record("1"), "test-rules", self.database_path)
        arguments = (
            self.database_path,
            self.companies_path,
            self.root / "snapshots",
            "training-v1",
        )
        create_snapshot(*arguments)

        with self.assertRaises(FileExistsError):
            create_snapshot(*arguments)


if __name__ == "__main__":
    unittest.main()
