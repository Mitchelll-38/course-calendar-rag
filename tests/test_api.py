"""Contract tests for the local synthetic API."""

import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from course_calendar_rag.api import app  # noqa: E402


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_identifies_synthetic_dataset(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "dataset": "synthetic"})

    def test_search_all_retrievers_return_ranked_synthetic_sources(self) -> None:
        fields = {"passage_id", "title", "section", "source", "score", "excerpt"}
        for retriever in ("bm25", "dense", "hybrid", "reranked"):
            with self.subTest(retriever=retriever):
                response = self.client.get(
                    "/search", params={"q": "COMP SCI 2C03", "retriever": retriever, "top_k": 2}
                )
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["query"], "COMP SCI 2C03")
                self.assertEqual(body["retriever"], retriever)
                self.assertEqual(len(body["sources"]), 2)
                self.assertEqual(body["sources"][0]["passage_id"], "course-compsci-2c03")
                self.assertEqual(
                    [item["score"] for item in body["sources"]],
                    sorted((item["score"] for item in body["sources"]), reverse=True),
                )
                for item in body["sources"]:
                    self.assertEqual(set(item), fields)
                    self.assertTrue(item["source"].startswith("synthetic://"))
                    self.assertTrue(item["excerpt"])

    def test_answer_is_extract_from_top_source_and_disclaimed(self) -> None:
        response = self.client.post("/answer", json={"question": "COMP SCI 2C03", "top_k": 2})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["retriever"], "hybrid")
        self.assertEqual(body["answer"], body["sources"][0]["excerpt"])
        self.assertEqual(body["sources"][0]["passage_id"], "course-compsci-2c03")
        self.assertIn("Synthetic", body["disclaimer"])

        for retriever in ("bm25", "dense", "hybrid", "reranked"):
            with self.subTest(retriever=retriever):
                response = self.client.post(
                    "/answer", json={"question": "COMP SCI 2C03", "retriever": retriever, "top_k": 1}
                )
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["retriever"], retriever)
                self.assertEqual(len(body["sources"]), 1)
                self.assertEqual(body["answer"], body["sources"][0]["excerpt"])

    def test_evaluation_all_retrievers_return_fixture_summary(self) -> None:
        for retriever in ("bm25", "dense", "hybrid", "reranked"):
            with self.subTest(retriever=retriever):
                response = self.client.get("/evaluation", params={"retriever": retriever})
                self.assertEqual(response.status_code, 200)
                body = response.json()
                self.assertEqual(body["retriever"], retriever)
                self.assertEqual(body["summary"]["question_count"], 20)
                self.assertIn("recall_at_5", body["summary"])
                self.assertIn("mrr", body["summary"])

    def test_cors_preflight_allows_only_local_frontends(self) -> None:
        for origin in ("http://localhost:3000", "http://127.0.0.1:3000"):
            with self.subTest(origin=origin):
                response = self.client.options(
                    "/answer",
                    headers={
                        "Origin": origin,
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    },
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["access-control-allow-origin"], origin)

        response = self.client.options(
            "/answer",
            headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_validation_rejects_bad_queries_modes_and_limits(self) -> None:
        for params in (
            {},
            {"q": "ab"},
            {"q": "   "},
            {"q": "x" * 501},
            {"q": " " + "x" * 500},
            {"q": "course load", "retriever": "unknown"},
            {"q": "course load", "top_k": 0},
            {"q": "course load", "top_k": 11},
        ):
            with self.subTest(params=params):
                self.assertEqual(self.client.get("/search", params=params).status_code, 422)

        for body in (
            {},
            {"question": "ab"},
            {"question": "   "},
            {"question": "x" * 501},
            {"question": " " + "x" * 500},
            {"question": "course load", "retriever": "unknown"},
            {"question": "course load", "top_k": 0},
            {"question": "course load", "top_k": 11},
        ):
            with self.subTest(body=body):
                self.assertEqual(self.client.post("/answer", json=body).status_code, 422)
        self.assertEqual(self.client.get("/evaluation?retriever=unknown").status_code, 422)


if __name__ == "__main__":
    unittest.main()
