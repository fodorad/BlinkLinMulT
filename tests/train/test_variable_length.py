"""Tests pinning LinMulT's variable-length behaviour.

The whole cross-corpus design rests on two claims about the sequence model, and
neither is obvious from its API — so both are asserted here rather than assumed:

1. **No time dimension is declared.** ``LinTConfig`` and ``LinMulTConfig`` have
   no ``time_dim`` field (``input_time_dim``/``output_time_dim`` exist only for
   the upsample/downsample heads, which this project does not use). One model
   instance therefore accepts any ``T``, before *and after* training, which is
   what lets one deployed model read 15, 24, 25, or 30 fps video.

2. **Padding is inert under a mask.** A window padded to a longer ``T`` with an
   all-``False`` tail produces the same output as the unpadded window. That is
   what makes a joint run over corpora of different frame rates correct, rather
   than merely runnable: the padding contributes nothing to either head.

Without (2), mixing RN15 and RN30 in one batch would silently feed the model
zero-frames as if they were real, and the shorter corpus's predictions would be
diluted by whatever the padding happened to encode.
"""

from __future__ import annotations

import unittest

import torch
from linmult import HeadConfig, LinMulT, LinMulTConfig, LinT, LinTConfig

FEATURE_DIM = 32
"""Width of the fake per-timestep features these tests feed the model."""

LENGTHS: tuple[int, ...] = (8, 13, 15, 30, 47)
"""Window lengths probed.

8 and 15 are what 0.5 s gives at 15 and 30 fps; 13 is 25 fps; the rest are
arbitrary, standing in for unconstrained in-the-wild video.
"""


def esr_head() -> HeadConfig:
    """A per-frame head, as eye state recognition uses.

    Returns:
        HeadConfig: A ``sequence`` head emitting one value per timestep.
    """
    return HeadConfig(type="sequence", name="esr", output_dim=1, hidden_dim=16)


def bpd_head() -> HeadConfig:
    """A per-window head, as blink presence detection uses.

    Returns:
        HeadConfig: A ``sequence_aggregation`` head emitting one value per clip.
    """
    return HeadConfig(type="sequence_aggregation", name="bpd", output_dim=1, hidden_dim=16)


def lint(*heads: HeadConfig) -> LinT:
    """Build a small single-stream model with the given heads.

    Args:
        *heads (HeadConfig): Heads to attach.

    Returns:
        LinT: The model, in eval mode.
    """
    model = LinT(
        LinTConfig(
            input_feature_dim=FEATURE_DIM,
            d_model=32,
            num_heads=8,
            cmt_num_layers=1,
            heads=list(heads),
        )
    )
    model.eval()
    return model


def sequence(batch: int, time: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A random input sequence and an all-valid mask.

    Args:
        batch (int): Batch size.
        time (int): Window length.

    Returns:
        tuple: ``(features, mask)``.
    """
    return torch.rand(batch, time, FEATURE_DIM), torch.ones(batch, time, dtype=torch.bool)


class TestNoDeclaredTimeDimension(unittest.TestCase):
    def test_the_config_has_no_time_dim_field(self):
        # If it did, one model could not span corpora of different rates.
        for config in (LinTConfig, LinMulTConfig):
            with self.subTest(config=config.__name__):
                self.assertFalse(hasattr(config(input_feature_dim=FEATURE_DIM), "time_dim"))

    def test_one_model_accepts_every_length(self):
        model = lint(esr_head())
        for time in LENGTHS:
            with self.subTest(time=time), torch.no_grad():
                output = model(*sequence(2, time))
                self.assertEqual(tuple(output["esr"].shape), (2, time, 1))

    def test_the_cross_modal_model_accepts_every_length(self):
        model = LinMulT(
            LinMulTConfig(
                input_feature_dim=[FEATURE_DIM, 16],
                d_model=32,
                num_heads=8,
                cmt_num_layers=1,
                branch_sat_num_layers=1,
                heads=[esr_head(), bpd_head()],
            )
        )
        model.eval()

        for time in LENGTHS:
            with self.subTest(time=time), torch.no_grad():
                images, mask = sequence(2, time)
                # The two streams carry independent masks, as the real batch does.
                features = torch.rand(2, time, 16)
                output = model([images, features], [mask, mask])
                self.assertEqual(tuple(output["esr"].shape), (2, time, 1))
                self.assertEqual(tuple(output["bpd"].shape), (2, 1))


class TestHeadShapes(unittest.TestCase):
    def test_the_sequence_head_emits_one_value_per_timestep(self):
        model = lint(esr_head())
        for time in LENGTHS:
            with self.subTest(time=time), torch.no_grad():
                self.assertEqual(model(*sequence(2, time))["esr"].shape[1], time)

    def test_the_aggregation_head_emits_one_value_per_window(self):
        model = lint(bpd_head())
        for time in LENGTHS:
            with self.subTest(time=time), torch.no_grad():
                # Collapsed over time whatever the length: a clip-level decision.
                self.assertEqual(tuple(model(*sequence(2, time))["bpd"].shape), (2, 1))

    def test_both_heads_coexist_on_one_model(self):
        # ESR and BPD are separate tasks over the same window, so they must be
        # expressible as two heads of one model rather than two models.
        model = lint(esr_head(), bpd_head())
        with torch.no_grad():
            output = model(*sequence(2, 15))
        self.assertEqual(set(output), {"esr", "bpd"})
        self.assertEqual(tuple(output["esr"].shape), (2, 15, 1))
        self.assertEqual(tuple(output["bpd"].shape), (2, 1))


class TestTrainedModelTransfers(unittest.TestCase):
    """A model trained at one length still runs at every other."""

    def train_briefly(self) -> LinT:
        """Take a few optimizer steps at a single window length.

        Returns:
            LinT: A model whose weights have actually moved, in eval mode.
        """
        torch.manual_seed(0)
        model = lint(esr_head(), bpd_head())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for _ in range(10):
            output = model(*sequence(4, 15))
            loss = output["esr"].mean() ** 2 + output["bpd"].mean() ** 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        return model

    def test_weights_trained_at_one_length_run_at_others(self):
        model = self.train_briefly()
        for time in LENGTHS:
            with self.subTest(time=time), torch.no_grad():
                output = model(*sequence(2, time))
                self.assertEqual(tuple(output["esr"].shape), (2, time, 1))
                self.assertEqual(tuple(output["bpd"].shape), (2, 1))
                self.assertTrue(torch.isfinite(output["bpd"]).all())


class TestPaddingIsInert(unittest.TestCase):
    """A masked pad must not change the answer.

    This is the property a joint run over 15 fps and 30 fps corpora depends on:
    the shorter corpus is padded to the longer's frame count, and its
    predictions must match what it would get unpadded.
    """

    def compare(self, model, real: int, padded: int) -> float:
        """Run one window natively and padded, and report the largest gap.

        Args:
            model: The model to run.
            real (int): Real frame count.
            padded (int): Length the window is padded to.

        Returns:
            float: Largest absolute difference across both heads.
        """
        torch.manual_seed(1)
        features = torch.rand(2, padded, FEATURE_DIM)

        mask = torch.zeros(2, padded, dtype=torch.bool)
        mask[:, :real] = True

        with torch.no_grad():
            padded_out = model(features, mask)
            native_out = model(features[:, :real], torch.ones(2, real, dtype=torch.bool))

        return max(
            (padded_out[key][..., :real, :] if key == "esr" else padded_out[key])
            .sub(native_out[key])
            .abs()
            .max()
            .item()
            for key in ("esr", "bpd")
        )

    def test_padding_does_not_change_the_prediction(self):
        model = lint(esr_head(), bpd_head())
        # 8 real frames padded to 15: exactly the RN15-in-a-RN30-run case.
        self.assertLess(self.compare(model, real=8, padded=15), 1e-4)

    def test_it_holds_across_pad_widths(self):
        model = lint(esr_head(), bpd_head())
        for real, padded in ((8, 15), (13, 15), (4, 30), (15, 47)):
            with self.subTest(real=real, padded=padded):
                self.assertLess(self.compare(model, real, padded), 1e-4)

    def test_it_holds_for_a_trained_model(self):
        # Untrained weights are near-symmetric and could mask a leak.
        torch.manual_seed(0)
        model = lint(esr_head(), bpd_head())
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(10):
            output = model(*sequence(4, 15))
            loss = output["esr"].pow(2).mean() + output["bpd"].pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

        self.assertLess(self.compare(model, real=8, padded=15), 1e-4)


if __name__ == "__main__":
    unittest.main()
