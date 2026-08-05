import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from course_calendar_rag.api import app  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_identifies_synthetic_dataset(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "dataset": "synthetic"})

    def test_search_returns_ranked_sources(self) -> None:
        response = self.client.get("/search", params={"q": "COMPSCI 2C03", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["sources"]), 2)

    def test_answer_is_grounded_and_disclaimed(self) -> None:
        response = self.client.post("/answer", json={"question": "What is a waitlist?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["sources"])
        self.assertIn("Synthetic", payload["disclaimer"])

    def test_invalid_top_k_is_rejected(self) -> None:
        self.assertEqual(self.client.get("/search", params={"q": "waitlist", "top_k": 50}).status_code, 422)

    def test_frontend_origin_is_allowed_by_cors(self) -> None:
        response = self.client.options(
            "/answer",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")


if __name__ == "__main__":
    unittest.main()
