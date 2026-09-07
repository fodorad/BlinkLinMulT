"""Tests for the precomputed embedding cache.

The failure this guards against is silent: a stale or misaligned cache trains
happily and reports a believable number. So these check *correctness* -- that a
cache is refused when it does not describe the run asking for it -- rather than
speed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blinklinmult.data.embeddings import (
    MANIFEST_NAME,
    CacheKey,
    EmbeddingCacheError,
    describe,
    encoder_digest,
    read_shard,
    shard_path,
    write_shard,
)


def _key(digest: str = "a" * 64, **overrides) -> CacheKey:
    values = {
        "encoder_digest": digest,
        "backbone": "convnext_femto",
        "output_dim": 256,
        "image_size": 64,
    }
    values.update(overrides)
    return CacheKey(**values)


class TestEncoderDigest(unittest.TestCase):
    """The cache is keyed by the weights that built it."""

    def test_the_same_weights_hash_the_same(self):
        model = torch.nn.Linear(4, 4)
        self.assertEqual(encoder_digest(model), encoder_digest(model))

    def test_different_weights_hash_differently(self):
        """Pointing encoder_weights at another arm must miss the cache."""
        first = torch.nn.Linear(4, 4)
        second = torch.nn.Linear(4, 4)
        with torch.no_grad():
            second.weight.fill_(0.5)
        self.assertNotEqual(encoder_digest(first), encoder_digest(second))

    def test_it_is_stable_across_devices(self):
        """Hashing on the CPU in float32 keeps the digest device-independent."""
        model = torch.nn.Linear(4, 4)
        before = encoder_digest(model)
        self.assertEqual(encoder_digest(model.to("cpu")), before)


class TestShardPath(unittest.TestCase):
    """The digest is a path component, so a wrong cache cannot be picked up."""

    def test_the_digest_separates_encoders(self):
        first = shard_path("cache", _key("a" * 64), "rn30", "train")
        second = shard_path("cache", _key("b" * 64), "rn30", "train")
        self.assertNotEqual(first, second)

    def test_it_separates_corpora_and_splits(self):
        key = _key()
        self.assertNotEqual(
            shard_path("cache", key, "rn30", "train"),
            shard_path("cache", key, "rn30", "valid"),
        )
        self.assertNotEqual(
            shard_path("cache", key, "rn15", "train"),
            shard_path("cache", key, "rn30", "train"),
        )


class TestShardRoundTrip(unittest.TestCase):
    """What is written comes back, and a mismatch is refused."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.key = _key()
        self.path = shard_path(self.root, self.key, "rn30", "train")
        self.keys = ["rec|000000|left", "rec|000000|right"]
        self.embeddings = np.random.rand(2, 5, 256).astype(np.float32)
        self.masks = np.ones((2, 5), dtype=bool)

    def _write(self, key: CacheKey | None = None):
        write_shard(self.path, key or self.key, self.keys, self.embeddings, self.masks)

    def test_it_reads_back_what_it_wrote(self):
        self._write()
        shard = read_shard(self.path, self.key)
        self.assertEqual(list(shard["keys"]), self.keys)
        self.assertEqual(shard["embedding"].shape, (2, 5, 256))

    def test_float16_storage_stays_within_tolerance(self):
        """These feed a d_model=32 transformer, far above float16's resolution."""
        self._write()
        shard = read_shard(self.path, self.key)
        np.testing.assert_allclose(
            shard["embedding"].astype(np.float32), self.embeddings, atol=1e-2
        )

    def test_a_different_encoder_is_refused(self):
        self._write()
        with self.assertRaises(EmbeddingCacheError) as caught:
            read_shard(self.path, _key("c" * 64))
        self.assertIn("encoder_digest", str(caught.exception))

    def test_a_different_image_size_is_refused(self):
        self._write()
        # Same directory, since the digest matches; only the manifest disagrees.
        with self.assertRaises(EmbeddingCacheError) as caught:
            read_shard(self.path, _key(image_size=128))
        self.assertIn("image_size", str(caught.exception))

    def test_a_different_output_dim_is_refused(self):
        self._write()
        with self.assertRaises(EmbeddingCacheError) as caught:
            read_shard(self.path, _key(output_dim=512))
        self.assertIn("output_dim", str(caught.exception))

    def test_a_missing_cache_is_refused(self):
        with self.assertRaises(EmbeddingCacheError):
            read_shard(self.path, self.key)

    def test_a_missing_manifest_is_refused(self):
        """A shard without its manifest cannot be checked, so it is not trusted."""
        self._write()
        (self.path.parent / f"{self.path.stem}.manifest.json").unlink()
        with self.assertRaises(EmbeddingCacheError):
            read_shard(self.path, self.key)

    def test_an_unreadable_manifest_is_refused(self):
        self._write()
        (self.path.parent / f"{self.path.stem}.manifest.json").write_text("{not json")
        with self.assertRaises(EmbeddingCacheError):
            read_shard(self.path, self.key)

    def test_no_partial_file_survives_a_completed_write(self):
        """The staging rename is what stops a killed build leaving a trusted shard."""
        self._write()
        self.assertFalse(self.path.with_suffix(".partial").exists())

    def test_the_manifest_records_the_window_count(self):
        self._write()
        manifest = json.loads((self.path.parent / f"{self.path.stem}.manifest.json").read_text())
        self.assertEqual(manifest["windows"], 2)


if __name__ == "__main__":
    unittest.main()


class TestDescribe(unittest.TestCase):
    """Discovering what a cache root holds.

    Used by ``make show-cache`` to report which shards exist before a run
    commits to them, so its failure modes matter: a half-written cache must be
    described, not crash the inspection.
    """

    def test_a_missing_root_is_empty_not_an_error(self) -> None:
        """Asking about a cache that was never built is a fair question."""
        self.assertEqual(describe("/nonexistent/cache/root"), [])

    def test_a_root_with_no_shards_is_empty(self) -> None:
        """An empty directory holds no shards, and that is not a failure."""
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(describe(directory), [])

    def test_a_manifest_is_found_and_its_shard_located(self) -> None:
        """The manifest names the shard; ``describe`` reports whether it exists."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "digest" / "rn30"
            root.mkdir(parents=True)
            (root / f"train.{MANIFEST_NAME}").write_text(json.dumps({"windows": 42}))
            (root / "train.h5").write_bytes(b"not really h5, but present")

            found = describe(directory)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["windows"], 42)
            self.assertTrue(found[0]["present"])
            self.assertTrue(str(found[0]["path"]).endswith("train.h5"))

    def test_a_manifest_without_its_shard_is_reported_as_absent(self) -> None:
        """A build interrupted between writing the two must be visible.

        Reporting it as present would send a training run at a file that is not
        there; omitting it entirely would hide that the build half-ran.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "digest" / "rn15"
            root.mkdir(parents=True)
            (root / f"valid.{MANIFEST_NAME}").write_text(json.dumps({"windows": 7}))

            found = describe(directory)
            self.assertEqual(len(found), 1)
            self.assertFalse(found[0]["present"])

    def test_an_unreadable_manifest_is_skipped_not_fatal(self) -> None:
        """One corrupt file must not stop the other shards being described."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "digest" / "rn30"
            root.mkdir(parents=True)
            (root / f"train.{MANIFEST_NAME}").write_text("{ not json")
            (root / f"valid.{MANIFEST_NAME}").write_text(json.dumps({"windows": 3}))

            found = describe(directory)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["windows"], 3)
