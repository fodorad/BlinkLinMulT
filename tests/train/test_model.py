"""Tests for the model architecture.

Shapes are checked with random tensors: the point is that a batch flows from the
loaded flat form through the backbone and the sequence model to per-timestep
logits, for every family and both tasks.

The backbones are constructed **untrained** (``backbone_pretrained=False``) so
the suite never downloads ImageNet weights; the pretrained path is a torchvision
concern, and what this project owns is the composition around it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from blinklinmult.data.schema import BLINK_PRESENCE, EYE_STATE
from blinklinmult.train.config import BACKBONES, ConfigError, ModelConfig
from blinklinmult.train.model import (
    DENSENET_CUT,
    MIN_IMAGE_SIZE,
    BlinkModel,
    EyeEncoder,
    ModelError,
    _widen_stem,
    build_backbone,
    build_model,
)

IMAGE_SIZE = 32
"""Small crops keep the suite fast; nothing here depends on the size."""

CROP_SIZE = 64
"""The real eye-crop size, where the 2x2/4x4 feature-map claims are meaningful."""


def config(**overrides) -> ModelConfig:
    defaults = {
        "family": "lint",
        "backbone": "densenet121",
        "backbone_pretrained": False,
        "backbone_output_dim": 16,
        "d_model": 16,
        "num_heads": 4,
        "cmt_num_layers": 1,
        "branch_sat_num_layers": 1,
        "head_hidden_dim": 8,
    }
    return ModelConfig(**{**defaults, **overrides})


def frozen_config(**overrides) -> ModelConfig:
    """A config with a frozen backbone, for the freeze-behaviour tests.

    ``ModelConfig`` rejects freeze + untrained, because freezing a
    randomly-initialised backbone can never learn — and that rejection is
    covered in ``test_config.py``. Here the untrained weights are irrelevant
    (freezing means they never move) and downloading real ImageNet weights just
    to check ``requires_grad`` would make the suite depend on the network, so
    the flag is set past the guard.
    """
    built = config(**overrides)
    object.__setattr__(built, "backbone_freeze", True)
    return built


def flat_images(batch: int = 2, time: int = 4, size: int = IMAGE_SIZE) -> torch.Tensor:
    """One eye's crops, as loaded: native (B, T, C, H, W) image form."""
    return torch.rand(batch, time, 3, size, size)


def mask(batch: int = 2, time: int = 4) -> torch.Tensor:
    return torch.ones(batch, time, dtype=torch.bool)


class TestBuildBackbone(unittest.TestCase):
    def test_every_supported_backbone_builds(self):
        for name in BACKBONES:
            with self.subTest(backbone=name):
                network, width = build_backbone(name, pretrained=False)
                self.assertIsInstance(network, torch.nn.Module)
                self.assertGreater(width, 0)

    def test_reported_width_matches_the_actual_output(self):
        # The width is read off the network rather than hard-coded, so a
        # library change cannot silently produce a wrong projection.
        for name in BACKBONES:
            for wide_stem in (False, True):
                with self.subTest(backbone=name, wide_stem=wide_stem):
                    network, width = build_backbone(name, pretrained=False, wide_stem=wide_stem)
                    network.eval()
                    with torch.no_grad():
                        output = network(torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE))
                    self.assertEqual(tuple(output.shape), (2, width))

    def test_densenet121_is_1024_dimensional(self):
        _, width = build_backbone("densenet121", pretrained=False)
        self.assertEqual(width, 1024)

    def test_the_default_backbone_is_supported(self):
        self.assertIn(ModelConfig().backbone, BACKBONES)

    def test_unknown_backbone_raises(self):
        with self.assertRaises(ModelError):
            build_backbone("vgg16", pretrained=False)


class TestWideStem(unittest.TestCase):
    """A 64px crop collapses to 2x2 on every ImageNet backbone.

    Four cells is very little to describe an eye, and the whole point of
    ``backbone_wide_stem`` is to buy 4x4 back. These tests pin the two claims
    the option rests on: that the map really does double, and that no pretrained
    parameter is dropped to achieve it.
    """

    def _map_size(self, network: torch.nn.Module) -> tuple[int, int]:
        """Spatial size of the map entering the final pooling layer."""
        pools = [m for m in network.modules() if isinstance(m, torch.nn.AdaptiveAvgPool2d)]
        seen: dict[str, tuple[int, ...]] = {}
        handle = pools[-1].register_forward_hook(
            lambda module, inputs, output: seen.update(shape=tuple(inputs[0].shape[2:]))
        )
        network.eval()
        with torch.no_grad():
            network(torch.rand(1, 3, CROP_SIZE, CROP_SIZE))
        handle.remove()
        return seen["shape"]

    def test_a_64px_crop_collapses_to_2x2_by_default(self):
        for name in ("mobilenetv4_conv_small", "shufflenet_v2", "resnet18", "densenet121"):
            with self.subTest(backbone=name):
                network, _ = build_backbone(name, pretrained=False, wide_stem=False)
                self.assertEqual(self._map_size(network), (2, 2))

    def test_the_wide_stem_doubles_the_map_to_4x4(self):
        for name in ("mobilenetv4_conv_small", "shufflenet_v2", "resnet18", "densenet121"):
            with self.subTest(backbone=name):
                network, _ = build_backbone(name, pretrained=False, wide_stem=True)
                self.assertEqual(self._map_size(network), (4, 4))

    def test_no_pretrained_parameter_is_dropped(self):
        # Both mechanisms are weight-preserving: a max-pool carries no
        # parameters, and re-striding a conv changes an attribute. The features
        # still shift -- see _widen_stem -- but nothing is discarded, so a
        # checkpoint stays loadable across the flag.
        for name in ("shufflenet_v2", "resnet18", "mobilenetv4_conv_small"):
            with self.subTest(backbone=name):
                stock, _ = build_backbone(name, pretrained=False, wide_stem=False)
                wide, _ = build_backbone(name, pretrained=False, wide_stem=True)
                self.assertEqual(
                    sum(p.numel() for p in stock.parameters()),
                    sum(p.numel() for p in wide.parameters()),
                )

    def test_the_width_is_unchanged_by_the_stem(self):
        for name in ("mobilenetv4_conv_small", "shufflenet_v2", "resnet18"):
            with self.subTest(backbone=name):
                _, stock = build_backbone(name, pretrained=False, wide_stem=False)
                _, wide = build_backbone(name, pretrained=False, wide_stem=True)
                self.assertEqual(stock, wide)

    def test_a_backbone_with_no_known_reduction_raises(self):
        with self.assertRaises(ModelError):
            _widen_stem("vgg16", torch.nn.Identity())


class TestEncoderWeights(unittest.TestCase):
    """A frame-wise checkpoint initialises a sequence model's encoder.

    The failure this guards against is silent: a checkpoint that does not fit
    would otherwise leave a randomly-initialised encoder that *looks*
    pretrained, and the run would report the transfer in its config while
    having gained nothing from it.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def checkpoint(self, **overrides) -> Path:
        """Write a Lightning-shaped checkpoint from a frame-wise model."""
        source = build_model(config(family="cnn", **overrides), [EYE_STATE], IMAGE_SIZE, None)
        state = {f"model.{k}": v for k, v in source.state_dict().items()}
        path = self.tmp / "frame_wise.ckpt"
        torch.save({"state_dict": state}, path)
        return path

    def test_the_encoder_weights_are_actually_copied(self):
        path = self.checkpoint()
        loaded = build_model(
            config(family="lint", encoder_weights=str(path)), [BLINK_PRESENCE], IMAGE_SIZE, None
        )
        source = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        for name, value in loaded.encoder.state_dict().items():
            key = f"model.encoder.{name}"
            if key in source:
                with self.subTest(tensor=name):
                    torch.testing.assert_close(value, source[key])

    def test_the_sequence_model_is_not_touched(self):
        # Only the encoder transfers; a frame-wise checkpoint has no transformer
        # to give, so the sequence model must still be freshly initialised.
        path = self.checkpoint()
        built = build_model(
            config(family="lint", encoder_weights=str(path)), [BLINK_PRESENCE], IMAGE_SIZE, None
        )
        self.assertTrue(any(p.requires_grad for p in built.sequence_model.parameters()))

    def test_the_encoder_stays_trainable(self):
        # Transferred as a starting point, not frozen: backbone_lr fine-tunes it.
        path = self.checkpoint()
        built = build_model(
            config(family="lint", encoder_weights=str(path)), [BLINK_PRESENCE], IMAGE_SIZE, None
        )
        self.assertTrue(all(p.requires_grad for p in built.encoder.parameters()))

    def test_a_missing_file_raises(self):
        with self.assertRaises(ModelError):
            build_model(
                config(family="lint", encoder_weights=str(self.tmp / "absent.ckpt")),
                [BLINK_PRESENCE],
                IMAGE_SIZE,
                None,
            )

    def test_a_checkpoint_without_encoder_tensors_raises(self):
        path = self.tmp / "wrong.ckpt"
        torch.save({"state_dict": {"something.else": torch.zeros(2)}}, path)
        with self.assertRaises(ModelError):
            build_model(
                config(family="lint", encoder_weights=str(path)),
                [BLINK_PRESENCE],
                IMAGE_SIZE,
                None,
            )

    def test_a_mismatched_backbone_raises_rather_than_loading_nothing(self):
        # The silent-failure case: densenet121 tensors cannot fill a
        # convnext_femto encoder, and quietly skipping them would leave the
        # encoder random while the config claims it was initialised.
        path = self.checkpoint(backbone="densenet121")
        with self.assertRaises(ModelError):
            build_model(
                config(family="lint", backbone="convnext_femto", encoder_weights=str(path)),
                [BLINK_PRESENCE],
                IMAGE_SIZE,
                None,
            )

    def test_no_weights_configured_leaves_the_encoder_alone(self):
        built = build_model(config(family="lint"), [BLINK_PRESENCE], IMAGE_SIZE, None)
        self.assertIsNotNone(built.encoder)


class TestTruncatedDenseNet(unittest.TestCase):
    """A smaller DenseNet has to be cut, not downloaded.

    121 is the smallest pretrained one that exists, so ``densenet121_truncated``
    drops the last dense block -- 31% of the parameters, spent on a 2x2 map at
    64px -- and keeps every weight before it.
    """

    def test_it_is_smaller_than_the_full_backbone(self):
        full, _ = build_backbone("densenet121", pretrained=False)
        cut, _ = build_backbone("densenet121_truncated", pretrained=False)
        self.assertLess(
            sum(p.numel() for p in cut.parameters()),
            sum(p.numel() for p in full.parameters()),
        )

    def test_the_reported_width_matches_the_output(self):
        # The width is probed off the truncated stack, not hard-coded: the final
        # transition halves the channels, so a literal would rot if the cut moved.
        network, width = build_backbone("densenet121_truncated", pretrained=False)
        network.eval()
        with torch.no_grad():
            output = network(torch.rand(2, 3, CROP_SIZE, CROP_SIZE))
        self.assertEqual(tuple(output.shape), (2, width))

    def test_the_last_dense_block_is_gone(self):
        cut, _ = build_backbone("densenet121_truncated", pretrained=False)
        self.assertNotIn(DENSENET_CUT, dict(cut.named_modules()))

    def test_it_still_accepts_the_wide_stem(self):
        network, width = build_backbone("densenet121_truncated", pretrained=False, wide_stem=True)
        network.eval()
        with torch.no_grad():
            output = network(torch.rand(2, 3, CROP_SIZE, CROP_SIZE))
        self.assertEqual(tuple(output.shape), (2, width))


class TestEyeEncoder(unittest.TestCase):
    def test_output_is_a_per_timestep_sequence(self):
        encoder = EyeEncoder(config(backbone_output_dim=16), IMAGE_SIZE)
        encoder.eval()
        with torch.no_grad():
            output = encoder(flat_images(2, 4))
        # One embedding per timestep: a sample carries a single eye.
        self.assertEqual(tuple(output.shape), (2, 4, 16))

    def test_output_dim_is_reported_correctly(self):
        encoder = EyeEncoder(config(backbone_output_dim=24), IMAGE_SIZE)
        self.assertEqual(encoder.output_dim, 24)

    def test_single_frame_window(self):
        encoder = EyeEncoder(config(backbone_output_dim=8), IMAGE_SIZE)
        encoder.eval()
        with torch.no_grad():
            output = encoder(flat_images(3, 1))
        self.assertEqual(tuple(output.shape), (3, 1, 8))

    def test_freezing_stops_backbone_gradients(self):
        encoder = EyeEncoder(frozen_config(), IMAGE_SIZE)
        self.assertFalse(any(p.requires_grad for p in encoder.backbone.parameters()))
        # The projection still trains.
        self.assertTrue(all(p.requires_grad for p in encoder.project.parameters()))

    def test_unfrozen_backbone_trains_by_default(self):
        encoder = EyeEncoder(config(), IMAGE_SIZE)
        self.assertTrue(all(p.requires_grad for p in encoder.backbone.parameters()))

    def test_a_crop_below_the_backbone_minimum_raises_readably(self):
        # Torch would otherwise report "Calculated output size: (512x0x0)" from
        # inside avg_pool2d, which says nothing about the crop being the cause.
        with self.assertRaises(ModelError) as ctx:
            EyeEncoder(config(), image_size=MIN_IMAGE_SIZE - 1)
        message = str(ctx.exception)
        self.assertIn(str(MIN_IMAGE_SIZE), message)
        self.assertIn("re-extract", message.lower())

    def test_the_minimum_crop_size_actually_works(self):
        encoder = EyeEncoder(config(backbone_output_dim=4), MIN_IMAGE_SIZE)
        encoder.eval()
        with torch.no_grad():
            output = encoder(flat_images(1, 2, MIN_IMAGE_SIZE))
        self.assertEqual(tuple(output.shape), (1, 2, 4))

    def test_normalisation_buffers_are_not_persisted(self):
        # They are constants, not learned state; persisting them would bloat
        # every checkpoint and break loading if the constants ever changed.
        encoder = EyeEncoder(config(), IMAGE_SIZE)
        self.assertNotIn("pixel_mean", encoder.state_dict())


class TestBuildModel(unittest.TestCase):
    def test_lint_family_builds_and_runs(self):
        model = build_model(config(family="lint"), [BLINK_PRESENCE], IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            output = model(flat_images(), mask())
        self.assertEqual(tuple(output[BLINK_PRESENCE].shape), (2, 4, 1))

    def test_linmult_family_builds_and_runs(self):
        model = build_model(
            config(family="linmult"), [BLINK_PRESENCE], IMAGE_SIZE, eye_feature_dim=12
        )
        model.eval()
        with torch.no_grad():
            output = model(flat_images(), mask(), torch.rand(2, 4, 12), mask())
        self.assertEqual(tuple(output[BLINK_PRESENCE].shape), (2, 4, 1))

    def test_cnn_family_builds_and_runs(self):
        model = build_model(config(family="cnn"), [EYE_STATE], IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            output = model(flat_images(), mask())
        self.assertEqual(tuple(output[EYE_STATE].shape), (2, 4, 1))

    def test_eye_state_head_emits_one_value_per_timestep(self):
        # One sample is one eye, so the head is width 1 like blink presence.
        model = build_model(config(family="lint"), [EYE_STATE], IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            output = model(flat_images(), mask())
        self.assertEqual(tuple(output[EYE_STATE].shape), (2, 4, 1))

    def test_joint_model_emits_both_targets(self):
        model = build_model(config(family="lint"), [BLINK_PRESENCE, EYE_STATE], IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            output = model(flat_images(), mask())
        self.assertEqual(set(output), {BLINK_PRESENCE, EYE_STATE})
        self.assertEqual(tuple(output[BLINK_PRESENCE].shape), (2, 4, 1))
        self.assertEqual(tuple(output[EYE_STATE].shape), (2, 4, 1))

    def test_both_targets_read_one_head(self):
        # The property the paper's architecture exists for. Two independent
        # heads could score a window "no frame shows a closed eye" AND "a blink
        # occurred" -- incoherent, and nothing in a two-head model forbids it.
        # Both targets reading one tensor makes that unrepresentable rather
        # than merely unlikely, so it is asserted as identity, not equality.
        for family, extra in (("lint", {}), ("cnn", {}), ("linmult", {"eye_feature_dim": 12})):
            with self.subTest(family=family):
                model = build_model(
                    config(family=family), [BLINK_PRESENCE, EYE_STATE], IMAGE_SIZE, **extra
                )
                model.eval()
                with torch.no_grad():
                    output = (
                        model(flat_images(), mask(), torch.rand(2, 4, 12), mask())
                        if family == "linmult"
                        else model(flat_images(), mask())
                    )
                self.assertIs(output[BLINK_PRESENCE], output[EYE_STATE])

    def test_one_head_is_built_however_many_targets(self):
        # A second target must not add parameters: it is a second reading of
        # the same sequence, not a second thing to learn.
        single = build_model(config(family="lint"), [EYE_STATE], IMAGE_SIZE)
        joint = build_model(config(family="lint"), [BLINK_PRESENCE, EYE_STATE], IMAGE_SIZE)
        self.assertEqual(
            sum(p.numel() for p in single.parameters()),
            sum(p.numel() for p in joint.parameters()),
        )

    def test_heads_are_keyed_by_target_name(self):
        # Looked up by name, never by position: the 1.x code indexed a dict
        # with [0], which is a KeyError rather than "the first head".
        model = build_model(config(family="lint"), [EYE_STATE], IMAGE_SIZE)
        model.eval()
        with torch.no_grad():
            output = model(flat_images(), mask())
        self.assertIn(EYE_STATE, output)

    def test_uses_eye_features_only_for_the_linmult_family(self):
        self.assertFalse(
            build_model(config(family="lint"), [EYE_STATE], IMAGE_SIZE).uses_eye_features
        )
        self.assertFalse(
            build_model(config(family="cnn"), [EYE_STATE], IMAGE_SIZE).uses_eye_features
        )
        self.assertTrue(
            build_model(
                config(family="linmult"), [EYE_STATE], IMAGE_SIZE, eye_feature_dim=8
            ).uses_eye_features
        )

    def test_linmult_without_eye_features_raises_at_construction(self):
        with self.assertRaises(ModelError) as ctx:
            build_model(config(family="linmult"), [EYE_STATE], IMAGE_SIZE)
        self.assertIn("family='lint'", str(ctx.exception))

    def test_linmult_called_without_its_second_modality_raises(self):
        model = build_model(config(family="linmult"), [EYE_STATE], IMAGE_SIZE, eye_feature_dim=8)
        with self.assertRaises(ModelError):
            model(flat_images(), mask())

    def test_unknown_target_has_no_head(self):
        with self.assertRaises(ModelError):
            build_model(config(family="lint"), ["gaze"], IMAGE_SIZE)


class TestGradientFlow(unittest.TestCase):
    def test_loss_reaches_the_backbone(self):
        model = build_model(config(family="lint"), [BLINK_PRESENCE], IMAGE_SIZE)
        output = model(flat_images(), mask())
        output[BLINK_PRESENCE].sum().backward()

        backbone_grads = [p.grad for p in model.encoder.backbone.parameters() if p.grad is not None]
        self.assertTrue(backbone_grads)
        self.assertTrue(any(g.abs().sum() > 0 for g in backbone_grads))

    def test_a_frozen_backbone_receives_no_gradient(self):
        model = build_model(frozen_config(family="lint"), [BLINK_PRESENCE], IMAGE_SIZE)
        output = model(flat_images(), mask())
        output[BLINK_PRESENCE].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.backbone.parameters()))


class TestAttentionVariants(unittest.TestCase):
    def test_softmax_and_linear_both_build(self):
        for attention in ("linear", "softmax"):
            with self.subTest(attention=attention):
                model = build_model(
                    config(family="lint", attention_type=attention),
                    [BLINK_PRESENCE],
                    IMAGE_SIZE,
                )
                model.eval()
                with torch.no_grad():
                    output = model(flat_images(), mask())
                self.assertEqual(tuple(output[BLINK_PRESENCE].shape), (2, 4, 1))


class TestMasking(unittest.TestCase):
    def test_a_partially_masked_window_runs(self):
        # A still-image corpus padded to the shared window: one real frame, the
        # rest masked out.
        model = build_model(config(family="lint"), [EYE_STATE], IMAGE_SIZE)
        partial = torch.zeros(2, 4, dtype=torch.bool)
        partial[:, 0] = True

        model.eval()
        with torch.no_grad():
            output = model(flat_images(2, 4), partial)
        self.assertEqual(tuple(output[EYE_STATE].shape), (2, 4, 1))
        self.assertTrue(torch.isfinite(output[EYE_STATE]).all())


if __name__ == "__main__":
    unittest.main()


class TestEncoderFreeze(unittest.TestCase):
    """A frozen encoder must hold still -- gradients *and* BatchNorm statistics."""

    def test_it_is_off_by_default(self):
        self.assertFalse(config().encoder_freeze)

    def test_freezing_without_weights_is_rejected(self):
        """Pinning an ImageNet encoder that never saw an eye is not an arm."""
        with self.assertRaises(ConfigError) as caught:
            config(encoder_freeze=True).validate()
        self.assertIn("encoder_weights", str(caught.exception))

    def test_an_unfrozen_encoder_trains(self):
        encoder = EyeEncoder(config(backbone_output_dim=16), IMAGE_SIZE)
        self.assertTrue(all(p.requires_grad for p in encoder.parameters()))
        encoder.train()
        self.assertTrue(encoder.training)

    def test_a_frozen_encoder_stays_in_eval_mode(self):
        """Lightning calls train() every epoch; BatchNorm would otherwise drift."""
        encoder = EyeEncoder(config(backbone_output_dim=16), IMAGE_SIZE)
        encoder.frozen = True
        encoder.train()
        self.assertFalse(encoder.training)
        encoder.train(True)
        self.assertFalse(encoder.training)

    def test_freezing_does_not_change_the_embedding(self):
        """The forward pass is unaffected; only learning is."""
        encoder = EyeEncoder(config(backbone_output_dim=16), IMAGE_SIZE)
        encoder.eval()
        images = torch.rand(2, 3, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            before = encoder(images)
        encoder.frozen = True
        encoder.train()
        with torch.no_grad():
            after = encoder(images)
        torch.testing.assert_close(before, after)


class TestEventHead(unittest.TestCase):
    """Blink intervals predicted *from* the ESR signal, not beside it.

    One shared head made the two annotations compete for the same parameters: a
    blink interval covers 3-4x more frames than closure, so a corpus annotating
    only intervals drags the closure prediction wider. Measured on the video
    benchmark, RN30's ESR F1 fell to 0.4107 against the frame-wise 0.6614.
    """

    def _model(self, **overrides):
        settings = {"backbone_output_dim": 16, "d_model": 8, "num_heads": 2}
        settings.update(overrides)
        return BlinkModel(
            config(family="lint", **settings),
            target_names=[EYE_STATE, BLINK_PRESENCE],
            image_size=IMAGE_SIZE,
        )

    def _batch(self):
        return (
            torch.rand(2, 6, 3, IMAGE_SIZE, IMAGE_SIZE),
            torch.ones(2, 6, dtype=torch.bool),
        )

    def test_no_event_head_keeps_one_shared_signal(self):
        """The default must be bit-identical to the previous behaviour."""
        model = self._model()
        self.assertIsNone(model.event_head)
        output = model(*self._batch())
        self.assertTrue(torch.equal(output[EYE_STATE], output[BLINK_PRESENCE]))

    def test_an_event_head_separates_the_two_targets(self):
        for kind in ("conv", "attention"):
            with self.subTest(kind=kind):
                output = self._model(event_head=kind)(*self._batch())
                self.assertFalse(torch.equal(output[EYE_STATE], output[BLINK_PRESENCE]))
                self.assertEqual(output[EYE_STATE].shape, output[BLINK_PRESENCE].shape)

    def test_detaching_blocks_the_event_gradient(self):
        """The whole point: closure is supervised by closure alone."""
        model = self._model(event_head="conv", event_head_detach=True)
        output = model(*self._batch())
        output[BLINK_PRESENCE].sum().backward()
        grads = [p.grad for p in model.sequence_model.parameters() if p.grad is not None]
        self.assertEqual(grads, [], "event loss reached the ESR trunk despite detach")

    def test_not_detaching_lets_it_through(self):
        """The alternative must be a real alternative, not a silent no-op."""
        model = self._model(event_head="conv", event_head_detach=False)
        output = model(*self._batch())
        output[BLINK_PRESENCE].sum().backward()
        grads = [p.grad for p in model.sequence_model.parameters() if p.grad is not None]
        self.assertNotEqual(grads, [])

    def test_the_event_head_sees_neighbouring_frames(self):
        """A pointwise head could only rescale; a blink is a *run* of frames."""
        model = self._model(event_head="conv").eval()
        images, mask = self._batch()
        with torch.no_grad():
            first = model(images, mask)[BLINK_PRESENCE]
            # Perturb one frame; a pointwise map would change only that column.
            images[:, 3] = torch.rand_like(images[:, 3])
            second = model(images, mask)[BLINK_PRESENCE]
        changed = (~torch.isclose(first, second, atol=1e-6)).squeeze(-1).any(0)
        self.assertGreater(int(changed.sum()), 1)

    def test_the_cnn_family_rejects_an_event_head(self):
        """A frame-wise model trains at T=1, so a temporal head has no context.

        `BlinkModel.forward` also returns from the cnn branch before the event
        head runs, so accepting the combination would silently do nothing.
        """
        with self.assertRaises(ConfigError) as caught:
            config(family="cnn", event_head="conv")
        self.assertIn("hysteresis", str(caught.exception))

    def test_it_is_absent_when_blink_presence_is_not_supervised(self):
        """An eye-state-only run has no interval annotation to train it."""
        model = BlinkModel(
            config(family="lint", backbone_output_dim=16, d_model=8, event_head="conv"),
            target_names=[EYE_STATE],
            image_size=IMAGE_SIZE,
        )
        self.assertIsNone(model.event_head)
