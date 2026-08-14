"""
Unit Tests for FastAPI REST & WebSocket Server.
"""

import unittest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
    HAS_TEST_CLIENT = True
except Exception:
    HAS_TEST_CLIENT = False


class TestAPIServer(unittest.TestCase):

    def setUp(self):
        if HAS_TEST_CLIENT:
            self.client = TestClient(app)
        else:
            self.client = None

    def test_root_endpoint(self):
        """Verify GET / root status endpoint."""
        if not HAS_TEST_CLIENT:
            self.skipTest("httpx not installed for TestClient")
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "online")

    def test_health_endpoint(self):
        """Verify GET /health healthcheck endpoint."""
        if not HAS_TEST_CLIENT:
            self.skipTest("httpx not installed for TestClient")
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "healthy")

    def test_predict_endpoint(self):
        """Verify POST /predict JSON risk estimation."""
        if not HAS_TEST_CLIENT:
            self.skipTest("httpx not installed for TestClient")
        payload = {
            "transcript_text": "Testing REST API endpoint.",
            "audio_feature": [120.0, 0.1] + [0.0] * 14,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        res_json = response.json()

        self.assertIn("stress_score", res_json)
        self.assertIn("stress_level", res_json)
        self.assertIn("fatigue_score", res_json)
        self.assertIn("confidence_score", res_json)


if __name__ == "__main__":
    unittest.main()

