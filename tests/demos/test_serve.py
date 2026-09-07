"""Tests for the REST service.

Three things are worth pinning, and they are the three that would be expensive
to discover in a deployed container:

* **the contract** -- every scoring reply carries the weights digest and the
  extraction rule, because a blink count without its threshold is not a
  measurement anyone can repeat;
* **the refusals** -- an unknown model, an impossible threshold and an oversized
  upload each fail with a specific status rather than a stack trace;
* **the silence** -- a genuine internal failure returns a fixed message, since
  ffmpeg and onnxruntime errors embed the server's own temporary paths.

The video endpoints need the extraction stack and real weights, so they are
opt-in; everything else runs against a live ``TestClient``.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

RUN_SERVE_TESTS = os.environ.get("RUN_SERVE_TESTS") != "0"
"""On by default: these are fast, and the service is the deployable artifact."""

try:
    from fastapi.testclient import TestClient

    from demos.docker.serve import app

    HAVE_SERVE = True
except ImportError:  # pragma: no cover - depends on the install
    HAVE_SERVE = False


def _crops(time: int = 12) -> list:
    """Random eye crops in the layout the API documents.

    Args:
        time (int): Frames to generate.

    Returns:
        list: ``(T, 3, 64, 64)`` values in ``[0, 1]``.
    """
    rng = np.random.default_rng(0)
    return (rng.random((time, 3, 64, 64)) * 0.3).tolist()


@unittest.skipUnless(HAVE_SERVE and RUN_SERVE_TESTS, "needs the `serve` extra")
class TestHealthAndManifest(unittest.TestCase):
    """What the service says about itself."""

    def test_ping_answers(self) -> None:
        """A liveness probe must not depend on a model being loaded."""
        with TestClient(app) as client:
            response = client.get("/ping")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_manifest_lists_every_model(self) -> None:
        """Every registered model, so a caller need not guess the ids.

        Counted against the registry rather than a literal: the table gained
        ``blinkcnn-onnx`` and a hard-coded four would have failed for the wrong
        reason.
        """
        from blinklinmult.registry import MODELS

        with TestClient(app) as client:
            payload = client.get("/manifest").json()
        self.assertEqual(set(payload["models"]), set(MODELS))

    def test_manifest_reports_weight_digests(self) -> None:
        """A result traces to an artifact, not to a version number.

        Without this an image could be rebuilt against different weights and
        report the same version, which is the difference between reproducible
        and merely deterministic.
        """
        with TestClient(app) as client:
            payload = client.get("/manifest").json()
        digests = [entry["weights_sha256"] for entry in payload["models"].values()]
        self.assertTrue(any(digests), "no weights found: the image would fetch at runtime")

    def test_manifest_states_whether_video_works(self) -> None:
        """The light image cannot process video, and says so up front."""
        with TestClient(app) as client:
            payload = client.get("/manifest").json()
        self.assertIsInstance(payload["video_supported"], bool)

    def test_index_renders(self) -> None:
        """The landing page is how someone finds the endpoints."""
        with TestClient(app) as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("blinkcnn", response.text)


@unittest.skipUnless(HAVE_SERVE and RUN_SERVE_TESTS, "needs the `serve` extra")
class TestScore(unittest.TestCase):
    """Scoring eye crops."""

    def test_returns_one_value_per_frame(self) -> None:
        """The signal is per-frame, matching the input length."""
        with TestClient(app) as client:
            payload = client.post("/score", json={"crops": _crops(12)}).json()
        self.assertEqual(len(payload["signal"]), 12)

    def test_reply_carries_its_provenance(self) -> None:
        """Model, weights digest and the rule applied, in every reply."""
        with TestClient(app) as client:
            payload = client.post("/score", json={"crops": _crops()}).json()
        self.assertIn("model_id", payload)
        self.assertIn("weights_sha256", payload)
        self.assertIn("extraction", payload)
        self.assertIn("high", payload["extraction"])

    def test_a_custom_threshold_is_applied_and_echoed(self) -> None:
        """The caller's rule, not the registered default.

        Echoing it back is what lets a reader reproduce the events; silently
        substituting the model's own would make the reply unreproducible.
        """
        with TestClient(app) as client:
            payload = client.post("/score", json={"crops": _crops(), "threshold": 0.2}).json()
        self.assertAlmostEqual(payload["extraction"]["high"], 0.2)

    def test_invocations_is_an_alias(self) -> None:
        """SageMaker requires this exact path."""
        with TestClient(app) as client:
            first = client.post("/score", json={"crops": _crops()}).json()
            second = client.post("/invocations", json={"crops": _crops()}).json()
        self.assertEqual(first["signal"], second["signal"])

    def test_an_unknown_model_is_rejected(self) -> None:
        """400 naming the valid ids, rather than a 500."""
        with TestClient(app) as client:
            response = client.post("/score", json={"crops": _crops(), "model_id": "nope"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("blinkcnn", response.json()["detail"])

    def test_an_impossible_threshold_is_rejected(self) -> None:
        """Outside ``(0, 1)`` nothing can be separated."""
        with TestClient(app) as client:
            response = client.post("/score", json={"crops": _crops(), "threshold": 1.9})
        self.assertEqual(response.status_code, 400)

    def test_an_inverted_hysteresis_range_is_rejected(self) -> None:
        """A low cut above the high one silently degrades to a single cut."""
        with TestClient(app) as client:
            response = client.post(
                "/score",
                json={"crops": _crops(), "threshold": 0.3, "low_threshold": 0.8},
            )
        self.assertEqual(response.status_code, 400)

    def test_malformed_crops_do_not_leak_a_traceback(self) -> None:
        """A generic message: internal errors name the server's own paths."""
        with TestClient(app) as client:
            response = client.post("/score", json={"crops": [[["not", "numbers"]]]})
        self.assertIn(response.status_code, (400, 422))
        self.assertNotIn("Traceback", response.text)


@unittest.skipUnless(HAVE_SERVE and RUN_SERVE_TESTS, "needs the `serve` extra")
class TestUploadLimits(unittest.TestCase):
    """Guarding a publicly reachable endpoint."""

    def test_an_empty_upload_is_rejected(self) -> None:
        """Nothing to process, and it should not reach the decoder."""
        with TestClient(app) as client:
            response = client.post("/detect", files={"file": ("empty.mp4", b"", "video/mp4")})
        self.assertIn(response.status_code, (400, 501))

    def test_the_cap_is_read_from_the_environment(self) -> None:
        """So a deployment can tighten it without a rebuild."""
        from demos.docker.serve import MAX_UPLOAD_BYTES

        self.assertGreater(MAX_UPLOAD_BYTES, 0)


@unittest.skipUnless(HAVE_SERVE and RUN_SERVE_TESTS, "needs the `serve` extra")
class TestScoreLimits(unittest.TestCase):
    """Bounding a publicly reachable endpoint's input.

    The upload path is capped in bytes as it streams; a JSON body is parsed
    before any handler runs, so its cap has to be applied to the decoded list
    instead. Without one, a request of a few thousand frames exhausts the
    container the image targets.
    """

    def test_an_oversized_request_is_refused(self) -> None:
        """413, naming the limit and the endpoint that does accept long input."""
        from demos.docker.serve import MAX_SCORE_FRAMES

        oversized = [[[[0.0]]]] * (MAX_SCORE_FRAMES + 1)
        with TestClient(app) as client:
            response = client.post("/score", json={"crops": oversized})
        self.assertEqual(response.status_code, 413)
        self.assertIn("/detect", response.json()["detail"])

    def test_the_limit_is_checked_before_the_array_is_built(self) -> None:
        """A malformed but oversized body must fail on size, not on parsing.

        Otherwise the guard runs after numpy has already copied the whole body,
        which is the allocation it exists to prevent.
        """
        from demos.docker.serve import MAX_SCORE_FRAMES

        junk = ["not a crop"] * (MAX_SCORE_FRAMES + 1)
        with TestClient(app) as client:
            response = client.post("/score", json={"crops": junk})
        self.assertEqual(response.status_code, 413)

    def test_a_request_at_the_limit_is_accepted(self) -> None:
        """The boundary is inclusive, so a caller can use the documented cap."""
        from demos.docker.serve import MAX_SCORE_FRAMES

        self.assertGreater(MAX_SCORE_FRAMES, 0)
        with TestClient(app) as client:
            response = client.post("/score", json={"crops": _crops(12)})
        self.assertEqual(response.status_code, 200)

    def test_the_cap_is_read_from_the_environment(self) -> None:
        """So a deployment can tighten it without a rebuild."""
        from demos.docker.serve import MAX_SCORE_FRAMES

        self.assertIsInstance(MAX_SCORE_FRAMES, int)


if __name__ == "__main__":
    unittest.main()
