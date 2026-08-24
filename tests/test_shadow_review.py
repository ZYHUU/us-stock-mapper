import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.database import (
    save_database_annotation,
    save_message,
    save_shadow_prediction,
)
from src.shadow_review import list_shadow_review_items, shadow_review_summary


LR_VERSION = "classic-lr-test"
LIGHTGBM_VERSION = "lightgbm-test"
MODEL_VERSIONS = [LR_VERSION, LIGHTGBM_VERSION]


class ShadowReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "test.db"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def sample_record(source_id: str = "shadow-1") -> dict[str, object]:
        return {
            "source_id": source_id,
            "source_type": "twitter",
            "source_url": f"https://example.com/{source_id}",
            "published_at": "2026-08-24 00:00:00",
            "text": "英伟达发布新芯片",
            "received_at": "2026-08-24T00:00:00+00:00",
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

    @staticmethod
    def candidate(score: float, predicted: bool) -> dict[str, object]:
        return {
            "canonical_code": "NASDAQ:NVDA",
            "mention": "英伟达",
            "match_type": "alias",
            "rule_confidence": 0.98,
            "model_score": score,
            "model_predicted": predicted,
        }

    def test_requires_all_models_before_adding_disagreement(self) -> None:
        message_id, _ = save_message(
            self.sample_record(), "rules-test", self.database_path
        )
        save_shadow_prediction(
            message_id,
            LR_VERSION,
            ["NASDAQ:NVDA"],
            [],
            [self.candidate(0.2, False)],
            self.database_path,
        )

        self.assertEqual(
            shadow_review_summary(MODEL_VERSIONS, self.database_path),
            {"total": 0, "reviewed": 0, "pending": 0},
        )
        self.assertEqual(
            list_shadow_review_items(MODEL_VERSIONS, self.database_path),
            [],
        )

    def test_consensus_disagreement_enters_queue_and_annotation_resolves_it(self) -> None:
        message_id, _ = save_message(
            self.sample_record(), "rules-test", self.database_path
        )
        for version, score in ((LR_VERSION, 0.2), (LIGHTGBM_VERSION, 0.3)):
            save_shadow_prediction(
                message_id,
                version,
                ["NASDAQ:NVDA"],
                [],
                [self.candidate(score, False)],
                self.database_path,
            )

        items = list_shadow_review_items(MODEL_VERSIONS, self.database_path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["predicted_codes"], ["NASDAQ:NVDA"])
        self.assertEqual(items[0]["priority"], 100)
        self.assertIn(
            "影子模型一致但与规则不同",
            items[0]["review_reasons"],
        )
        self.assertEqual(
            shadow_review_summary(MODEL_VERSIONS, self.database_path),
            {"total": 1, "reviewed": 0, "pending": 1},
        )

        save_database_annotation(
            key="id:shadow-1",
            correct_codes=["NASDAQ:NVDA"],
            scope_codes=["NASDAQ:NVDA"],
            annotator="shadow_review",
            confidence="high",
            path=self.database_path,
        )

        self.assertEqual(
            list_shadow_review_items(MODEL_VERSIONS, self.database_path),
            [],
        )
        self.assertEqual(
            shadow_review_summary(MODEL_VERSIONS, self.database_path),
            {"total": 1, "reviewed": 1, "pending": 0},
        )
        reviewed_items = list_shadow_review_items(
            MODEL_VERSIONS,
            self.database_path,
            unreviewed_only=False,
        )
        self.assertEqual(reviewed_items[0]["annotation"]["annotator"], "shadow_review")


if __name__ == "__main__":
    unittest.main()
