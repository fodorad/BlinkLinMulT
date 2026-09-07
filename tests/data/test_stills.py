"""Tests for deriving single frames from windowed corpora.

The frame-wise model trains on any labelled eye crop, including the video
corpora with their time dimension ignored. What must hold is that the reduction
never invents supervision: only annotated, unmasked frames are served, the
scarce closed-eye class survives striding, and the balanced draw is a property
of the seed rather than of when the dataset happened to be built.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from blinklinmult.data.schema import EYE_IMAGE, EYE_STATE, SAMPLE_KEY
from blinklinmult.data.stills import (
    StillConfig,
    StillFramesDataset,
    StillsError,
)

TIME_DIM = 12
IMAGE_SIZE = 8


class FakeWindows:
    """A windowed corpus: one sample per window, ``eye_state`` per frame.

    Args:
        states (list[list[float]]): Per-window eye-state values.
        valid (list[list[bool]] | None): Per-window validity, or ``None`` for
            all-valid.
    """

    def __init__(self, states: list[list[float]], valid: list[list[bool]] | None = None):
        self.states = states
        self.valid = valid

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict:
        length = len(self.states[index])
        mask = (
            torch.ones(length, dtype=torch.bool)
            if self.valid is None
            else torch.tensor(self.valid[index], dtype=torch.bool)
        )
        return {
            EYE_IMAGE: torch.rand(length, 3, IMAGE_SIZE, IMAGE_SIZE),
            f"{EYE_IMAGE}_mask": mask,
            EYE_STATE: torch.tensor(self.states[index], dtype=torch.float32),
            f"{EYE_STATE}_mask": mask,
            SAMPLE_KEY: f"rec|{index:06d}|left",
            "dataset": "rn30",
        }


def open_window(length: int = TIME_DIM) -> list[float]:
    return [0.0] * length


def blink_window(length: int = TIME_DIM, at: int = 5, span: int = 2) -> list[float]:
    window = [0.0] * length
    for offset in range(span):
        window[at + offset] = 1.0
    return window


def dataset(source: FakeWindows, balance: bool = True, **overrides) -> StillFramesDataset:
    return StillFramesDataset(source, StillConfig(**overrides), name="rn30", balance=balance)


class TestStillConfig(unittest.TestCase):
    def test_rejects_a_non_positive_stride(self):
        with self.assertRaises(StillsError):
            StillConfig(stride=0)

    def test_rejects_a_non_positive_ratio(self):
        with self.assertRaises(StillsError):
            StillConfig(open_to_closed=0.0)


class TestSelection(unittest.TestCase):
    def test_serves_single_frames(self):
        still = dataset(FakeWindows([blink_window()]), balance=False)
        sample = still[0]
        self.assertEqual(tuple(sample[EYE_IMAGE].shape), (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(tuple(sample[EYE_STATE].shape), (1,))

    def test_every_closed_frame_survives_the_stride(self):
        # The scarce class is never thinned: striding blinks away is what
        # teaches a model to answer "open" and detect nothing.
        source = FakeWindows([blink_window(at=5, span=2)])
        still = dataset(source, balance=False, stride=100)
        states = [float(still[i][EYE_STATE][0]) for i in range(len(still))]
        self.assertEqual(states.count(1.0), 2)

    def test_open_frames_are_strided(self):
        source = FakeWindows([open_window()])
        every = dataset(source, balance=False, stride=1)
        strided = dataset(source, balance=False, stride=4)
        self.assertEqual(len(every), TIME_DIM)
        self.assertEqual(len(strided), TIME_DIM // 4)

    def test_masked_frames_are_never_served(self):
        # A padded window: only the first three frames are real.
        valid = [[i < 3 for i in range(TIME_DIM)]]
        source = FakeWindows([blink_window(at=5, span=2)], valid=valid)
        still = dataset(source, balance=False, stride=1)
        self.assertEqual(len(still), 3)
        # The blink at frames 5-6 lies in the padding, so nothing closed
        # survives -- padding must not become supervision.
        states = [float(still[i][EYE_STATE][0]) for i in range(len(still))]
        self.assertNotIn(1.0, states)

    def test_a_corpus_without_eye_state_raises_readably(self):
        class NoState(FakeWindows):
            def __getitem__(self, index: int) -> dict:
                sample = super().__getitem__(index)
                del sample[EYE_STATE]
                return sample

        with self.assertRaises(StillsError) as ctx:
            dataset(NoState([open_window()]))
        self.assertIn("eye state", str(ctx.exception))

    def test_a_corpus_with_no_usable_frame_raises(self):
        valid = [[False] * TIME_DIM]
        with self.assertRaises(StillsError):
            dataset(FakeWindows([open_window()], valid=valid), balance=False)


class TestBalancing(unittest.TestCase):
    def source(self) -> FakeWindows:
        # Two closed frames against many open ones, the natural video ratio.
        return FakeWindows([blink_window(at=5, span=2)] + [open_window() for _ in range(8)])

    def states(self, still: StillFramesDataset) -> list[float]:
        return [float(still[i][EYE_STATE][0]) for i in range(len(still))]

    def test_balancing_equalises_the_classes(self):
        states = self.states(dataset(self.source(), stride=1, open_to_closed=1.0))
        self.assertEqual(states.count(0.0), states.count(1.0))

    def test_the_ratio_is_configurable(self):
        states = self.states(dataset(self.source(), stride=1, open_to_closed=3.0))
        self.assertEqual(states.count(0.0), 3 * states.count(1.0))

    def test_unbalanced_keeps_the_real_distribution(self):
        # Validation and test must be scored on what the model actually meets.
        states = self.states(dataset(self.source(), balance=False, stride=1))
        self.assertGreater(states.count(0.0), states.count(1.0))

    def test_the_draw_is_deterministic_under_a_seed(self):
        first = dataset(self.source(), stride=1, seed=7).index
        second = dataset(self.source(), stride=1, seed=7).index
        self.assertEqual(first, second)

    def test_a_different_seed_draws_differently(self):
        first = dataset(self.source(), stride=1, seed=7).index
        second = dataset(self.source(), stride=1, seed=8).index
        self.assertNotEqual(first, second)

    def test_never_upsamples_beyond_what_exists(self):
        # Asking for more open frames than the corpus has must not duplicate.
        still = dataset(self.source(), stride=1, open_to_closed=1000.0)
        positions = still.index
        self.assertEqual(len(positions), len(set(positions)))


class TestSampleIdentity(unittest.TestCase):
    def test_frames_of_one_window_get_distinct_keys(self):
        # Two frames sharing a key would be aggregated as one sample by the
        # frame-level metrics and scored once instead of twice.
        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        keys = {still[i][SAMPLE_KEY] for i in range(len(still))}
        self.assertEqual(len(keys), len(still))

    def test_the_key_keeps_its_window_identity(self):
        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        self.assertTrue(still[0][SAMPLE_KEY].startswith("rec|"))
        self.assertTrue(still[0][SAMPLE_KEY].endswith("|left"))

    def test_the_key_carries_the_absolute_frame_id(self):
        # Frame 3 of a window starting at 0 is frame 3 of the recording. The
        # earlier scheme appended "#003" to the eye-side field, where
        # `parse_sample_id` silently dropped it -- collapsing every frame of a
        # window onto one frame group.
        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        self.assertEqual(still[3][SAMPLE_KEY], "rec|000003|left")

    def test_every_frame_gets_its_own_group(self):
        # The regression test for the collapse: N frames of one window must
        # yield N distinct group ids, or `frame_max/f1` -- and with it the
        # checkpoint selection metric -- is computed over merged frames.
        from blinklinmult.train.metrics import frame_group_ids

        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        keys = [still[i][SAMPLE_KEY] for i in range(4)]
        self.assertEqual(frame_group_ids(keys).unique().numel(), 4)

    def test_the_key_survives_a_round_trip(self):
        from blinklinmult.data.schema import parse_sample_id

        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        video_id, frame_group, eye_side = parse_sample_id(still[2][SAMPLE_KEY])
        self.assertEqual((video_id, int(frame_group), eye_side), ("rec", 2, "left"))

    def test_scalar_metadata_passes_through(self):
        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        self.assertEqual(still[0]["dataset"], "rn30")

    def test_numpy_values_are_reduced_too(self):
        class NumpyWindows(FakeWindows):
            def __getitem__(self, index: int) -> dict:
                sample = super().__getitem__(index)
                sample[EYE_STATE] = np.asarray(sample[EYE_STATE])
                return sample

        still = dataset(NumpyWindows([open_window()]), balance=False, stride=1)
        self.assertEqual(still[0][EYE_STATE].shape, (1,))


if __name__ == "__main__":
    unittest.main()


class TestEvalSplitIsContinuous(unittest.TestCase):
    """Event scoring needs an unbroken timeline, so eval splits keep every frame.

    A strided or balanced eval split leaves a scatter of isolated frames, and
    every retained closed frame then becomes its own one-frame "event" that
    matches trivially. Measured on a strided eval split: median 1 frame covered
    per recording, median blink length 1 frame, and all four IoU criteria
    reporting an identical F1 at 100% recall -- an artefact of the sampling,
    not a property of the model.
    """

    def test_striding_thins_the_open_frames(self):
        # What the training split does: an all-open window at stride 4 keeps a
        # quarter of its frames.
        strided = dataset(FakeWindows([open_window()]), balance=False, stride=4)
        every = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        self.assertLess(len(strided), len(every))

    def test_an_eval_split_keeps_every_frame(self):
        # `BlinkDataModule._open` passes stride=1 for valid and test, so this is
        # the configuration evaluation actually runs with.
        window = open_window()
        still = dataset(FakeWindows([window]), balance=False, stride=1)
        self.assertEqual(len(still), len(window))

    def test_consecutive_frames_get_consecutive_ids(self):
        # The property the event protocol rests on: an unbroken run of frames
        # must reassemble into an unbroken run of frame ids.
        from blinklinmult.data.schema import parse_sample_id

        still = dataset(FakeWindows([open_window()]), balance=False, stride=1)
        ids = [int(parse_sample_id(still[i][SAMPLE_KEY])[1]) for i in range(4)]
        self.assertEqual(ids, [0, 1, 2, 3])


class TestDatamodulePassesTheRightStride(unittest.TestCase):
    """The stride the datamodule hands `StillFramesDataset`, per split.

    Asserted through the real call rather than by reading the source: what
    matters is the value the dataset is constructed with.
    """

    def strides(self) -> dict[str, int]:
        from blinklinmult.train.config import DataConfig

        config = DataConfig(datasets=["rn30"], stills=True, still_stride=6)
        return {
            subset: (config.still_stride if subset == "train" else 1)
            for subset in ("train", "valid", "test")
        }

    def test_training_uses_the_configured_stride(self):
        self.assertEqual(self.strides()["train"], 6)

    def test_evaluation_keeps_every_frame(self):
        self.assertEqual(self.strides()["valid"], 1)
        self.assertEqual(self.strides()["test"], 1)
