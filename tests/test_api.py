import io
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from app import app


def audio_file(samples=None):
    buffer = io.BytesIO()
    wavfile.write(buffer, 8000, np.ones(400, dtype=np.int16) if samples is None else samples)
    buffer.seek(0)
    return buffer, "test.wav"


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_invalid_trace_parameters(self):
        for query in ("rays=nan", "rays=0", "rays=100.5", "max_time=inf", "sx=nan&sy=1", "sx=1"):
            with self.subTest(query=query):
                response = self.client.get("/api/trace/cathedral?" + query)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.json)

    def test_invalid_positions(self):
        for query in ("sx=-1&sy=1", "sx=8&sy=1.5&rx=8&ry=1.5"):
            self.assertEqual(self.client.get("/api/trace/cathedral?" + query).status_code, 400)

    def test_missing_or_corrupt_audio(self):
        self.assertEqual(self.client.post("/api/process").status_code, 400)
        response = self.client.post("/api/process", data={"audio": (io.BytesIO(b"bad"), "a.wav")})
        self.assertEqual(response.status_code, 400)

    def test_empty_audio(self):
        response = self.client.post("/api/process", data={"audio": audio_file(np.array([], dtype=np.int16))})
        self.assertEqual(response.status_code, 400)

    def test_large_upload(self):
        with patch.dict(app.config, MAX_CONTENT_LENGTH=16):
            self.assertEqual(self.client.post("/api/process", data={"audio": audio_file()}).status_code, 413)


if __name__ == "__main__":
    unittest.main()
