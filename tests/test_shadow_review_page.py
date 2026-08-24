import unittest
from pathlib import Path


PAGE_PATH = Path(__file__).resolve().parents[1] / "static" / "shadow_review.html"


class ShadowReviewPageTest(unittest.TestCase):
    def test_model_action_saves_prediction_instead_of_only_selecting_it(self) -> None:
        page = PAGE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "button.onclick = () => save(prediction.model_codes);",
            page,
        )
        self.assertNotIn(
            "button.onclick = () => setSelectedCodes(prediction.model_codes);",
            page,
        )


if __name__ == "__main__":
    unittest.main()
