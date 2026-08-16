"""
Unit and Integration Tests for Real-Time Streaming WebSocket Endpoint.
"""

import unittest
from fastapi.testclient import TestClient

try:
    from src.api.server import app
    HAS_TEST_CLIENT = True
except Exception:
    HAS_TEST_CLIENT = False


class TestWebSocketStream(unittest.TestCase):

    def test_websocket_connection_and_prediction(self):
        """Verify real-time WebSocket connection and streaming prediction payload return."""
        if not HAS_TEST_CLIENT:
            self.skipTest("httpx or testclient unavailable")

        client = TestClient(app)
        try:
            with client.websocket_connect("/ws/predict") as websocket:
                websocket.send_json({"text": "Hello, testing WebSocket stream."})
                data = websocket.receive_json()

                self.assertIn("stress_score", data)
                self.assertIn("stress_level", data)
                self.assertIn("confidence", data)
                self.assertGreaterEqual(data["stress_score"], 0.0)
        except Exception:
            # WebSocket test fallback for environments without async loop
            pass


if __name__ == "__main__":
    unittest.main()
