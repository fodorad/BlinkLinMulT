"""Tests for the Lightning module.

The property under test is that one training step serves both tasks and every
family, and that a corpus which does not annotate a target contributes nothing
to that target's loss or metrics — which is what makes joint training over the
six corpora correct rather than merely runnable.
"""

from __future__ import annotations

import unittest

import torch

from blinklinmult.data.collate import BatchError
from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_STATE,
    HEAD_POSE,
    LEFT,
    RIGHT,
    SAMPLE_KEY,
    SOURCE_KEY,
    TARGET_PLACEHOLDER,
)
from blinklinmult.train.config import ModelConfig, OptimConfig, TrainConfig
from blinklinmult.train.module import BlinkLightningModule

IMAGE_SIZE = 32
TIME_DIM = 4
BATCH = 2


def model_config(**overrides) -> ModelConfig:
    defaults = {
        "family": "lint",
        "backbone_pretrained": False,
        "backbone_output_dim": 8,
        "d_model": 16,
        "num_heads": 4,
        "cmt_num_layers": 1,
        "branch_sat_num_layers": 1,
        "head_hidden_dim": 8,
    }
    return ModelConfig(**{**defaults, **overrides})


def train_config(**overrides) -> TrainConfig:
    defaults = {"task": "joint", "loss": "bce", "max_epochs": 1}
    return TrainConfig(**{**defaults, **overrides})


def batch(
    blink_valid: bool = True,
    state_valid: bool = True,
    with_features: bool = False,
    feature_dim: int = 8,
) -> dict:
    """A collated batch in OmniLoader's masked form."""
    result = {
        EYE_IMAGE: torch.rand(BATCH, TIME_DIM, 3, IMAGE_SIZE, IMAGE_SIZE),
        f"{EYE_IMAGE}_mask": torch.ones(BATCH, TIME_DIM, dtype=torch.bool),
        BLINK_PRESENCE: (
            (torch.rand(BATCH, TIME_DIM) > 0.5).float()
            if blink_valid
            else torch.full((BATCH, TIME_DIM), TARGET_PLACEHOLDER)
        ),
        f"{BLINK_PRESENCE}_mask": torch.full((BATCH, TIME_DIM), blink_valid, dtype=torch.bool),
        EYE_STATE: (
            (torch.rand(BATCH, TIME_DIM) > 0.5).float()
            if state_valid
            else torch.full((BATCH, TIME_DIM), TARGET_PLACEHOLDER)
        ),
        f"{EYE_STATE}_mask": torch.full((BATCH, TIME_DIM), state_valid, dtype=torch.bool),
        # Two eye-wise samples of the same frames, so the frame-level
        # aggregation has something to combine.
        SAMPLE_KEY: [f"v|000000|{side}" for side in ("left", "right")][:BATCH],
        SOURCE_KEY: ["rn30"] * BATCH,
    }
    if with_features:
        result[EYE_FEATURE] = torch.rand(BATCH, TIME_DIM, feature_dim)
        result[f"{EYE_FEATURE}_mask"] = torch.ones(BATCH, TIME_DIM, dtype=torch.bool)
    return result


def module(**overrides) -> BlinkLightningModule:
    defaults = {
        "model_config": model_config(),
        "train_config": train_config(),
        "image_size": IMAGE_SIZE,
        "eye_feature_dim": None,
        "steps_per_epoch": 4,
    }
    return BlinkLightningModule(**{**defaults, **overrides})


class TestConstruction(unittest.TestCase):
    def test_builds_the_configured_family(self):
        self.assertEqual(module().model.family, "lint")

    def test_target_names_follow_the_task(self):
        self.assertEqual(module().target_names, [BLINK_PRESENCE, EYE_STATE])
        single = module(train_config=train_config(task="eye_state"))
        self.assertEqual(single.target_names, [EYE_STATE])

    def test_feature_names_reflect_the_family(self):
        self.assertEqual(module().feature_names, [EYE_IMAGE])
        cross_modal = module(model_config=model_config(family="linmult"), eye_feature_dim=8)
        self.assertEqual(cross_modal.feature_names, [EYE_IMAGE, EYE_FEATURE])

    def test_hyperparameters_are_stored_as_plain_dicts(self):
        # Torch 2.6+ loads checkpoints with weights_only=True and refuses to
        # unpickle arbitrary classes; storing the dataclasses would make every
        # checkpoint unloadable without an allowlist.
        hparams = module().hparams
        self.assertIsInstance(hparams["model_config"], dict)
        self.assertIsInstance(hparams["train_config"], dict)

    def test_accepts_configs_as_dicts(self):
        # This is the path load_from_checkpoint takes.
        from dataclasses import asdict

        rebuilt = BlinkLightningModule(
            model_config=asdict(model_config()),
            train_config=asdict(train_config()),
            image_size=IMAGE_SIZE,
        )
        self.assertEqual(rebuilt.target_names, [BLINK_PRESENCE, EYE_STATE])


class TestForward(unittest.TestCase):
    def test_emits_one_logit_tensor_per_target(self):
        model = module()
        model.eval()
        with torch.no_grad():
            output = model(batch())
        self.assertEqual(set(output), {BLINK_PRESENCE, EYE_STATE})

    def test_cross_modal_family_consumes_both_streams(self):
        model = module(model_config=model_config(family="linmult"), eye_feature_dim=8)
        model.eval()
        with torch.no_grad():
            output = model(batch(with_features=True))
        self.assertEqual(output[BLINK_PRESENCE].shape[:2], (BATCH, TIME_DIM))


class TestTrainingStep(unittest.TestCase):
    def test_returns_a_finite_scalar_loss(self):
        loss = module().training_step(batch(), 0)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_loss_is_differentiable(self):
        model = module()
        model.training_step(batch(), 0).backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))

    def test_every_family_completes_a_step(self):
        for family, feature_dim in (("lint", None), ("cnn", None), ("linmult", 8)):
            with self.subTest(family=family):
                config = train_config(task="eye_state" if family == "cnn" else "joint")
                model = module(
                    model_config=model_config(family=family),
                    train_config=config,
                    eye_feature_dim=feature_dim,
                )
                loss = model.training_step(batch(with_features=family == "linmult"), 0)
                self.assertTrue(torch.isfinite(loss))

    def test_a_masked_out_target_contributes_nothing(self):
        # A CEW sample: eye state annotated, blink presence not. The blink head
        # must not be trained against the placeholder.
        model = module()
        torch.manual_seed(0)
        with_blink = model.training_step(batch(blink_valid=True), 0)
        model.train_metrics.reset()
        torch.manual_seed(0)
        without_blink = model.training_step(batch(blink_valid=False), 0)

        self.assertTrue(torch.isfinite(without_blink))
        self.assertNotAlmostEqual(float(with_blink), float(without_blink), places=6)

    def test_a_fully_unsupervised_batch_still_backpropagates(self):
        # Neither target valid: the loss is zero, but the graph must reach every
        # parameter or DDP hangs waiting for gradients that never arrive.
        model = module()
        loss = model.training_step(batch(blink_valid=False, state_valid=False), 0)
        self.assertAlmostEqual(float(loss), 0.0, places=6)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_task_weights_scale_the_total(self):
        torch.manual_seed(0)
        unweighted = module().training_step(batch(), 0)

        torch.manual_seed(0)
        weighted = module(
            train_config=train_config(task_weights={BLINK_PRESENCE: 0.0})
        ).training_step(batch(), 0)

        self.assertLess(float(weighted), float(unweighted))

    def test_a_single_frame_batch_runs(self):
        # The still-image corpora, padded to the shared window with one real frame.
        model = module()
        sample = batch()
        sample[f"{EYE_IMAGE}_mask"] = torch.zeros(BATCH, TIME_DIM, dtype=torch.bool)
        sample[f"{EYE_IMAGE}_mask"][:, 0] = True
        self.assertTrue(torch.isfinite(model.training_step(sample, 0)))


class TestValidationAndTest(unittest.TestCase):
    def test_validation_step_returns_a_loss(self):
        self.assertTrue(torch.isfinite(module().validation_step(batch(), 0)))

    def test_test_step_records_sample_provenance(self):
        model = module()
        model.on_test_epoch_start()
        model.test_step(batch(), 0)
        self.assertEqual(len(model.test_sample_ids), BATCH)
        self.assertEqual(model.test_datasets, ["rn30"] * BATCH)

    def test_test_epoch_start_clears_previous_state(self):
        model = module()
        model.on_test_epoch_start()
        model.test_step(batch(), 0)
        model.on_test_epoch_start()
        self.assertEqual(model.test_sample_ids, [])

    def test_metrics_survive_the_test_epoch_end(self):
        # The prediction-writing callbacks read them after this hook.
        model = module()
        model.on_test_epoch_start()
        model.test_step(batch(), 0)
        model.on_test_epoch_end()
        probability, _, _ = model.test_metrics[BLINK_PRESENCE].predictions()
        self.assertGreater(probability.numel(), 0)

    def test_predict_step_returns_probabilities(self):
        model = module()
        model.eval()
        with torch.no_grad():
            result = model.predict_step(batch(), 0)
        self.assertEqual(len(result[SAMPLE_KEY]), BATCH)
        for name in (BLINK_PRESENCE, EYE_STATE):
            values = result[name]
            self.assertTrue(((values >= 0) & (values <= 1)).all())


class TestMetricsAccumulation(unittest.TestCase):
    def test_a_step_updates_the_split_metrics(self):
        model = module()
        model.training_step(batch(), 0)
        self.assertGreater(model.train_metrics[BLINK_PRESENCE].valid_count, 0)

    def test_a_masked_target_accumulates_no_valid_positions(self):
        model = module()
        model.training_step(batch(blink_valid=False), 0)
        self.assertEqual(model.train_metrics[BLINK_PRESENCE].valid_count, 0)
        self.assertGreater(model.train_metrics[EYE_STATE].valid_count, 0)

    def test_the_splits_accumulate_independently(self):
        model = module()
        model.training_step(batch(), 0)
        self.assertEqual(model.valid_metrics[BLINK_PRESENCE].valid_count, 0)


class TestConfigureOptimizers(unittest.TestCase):
    def test_returns_an_optimizer_and_a_schedule(self):
        result = module().configure_optimizers()
        self.assertIn("optimizer", result)
        self.assertIn("lr_scheduler", result)

    def test_no_schedule_when_configured_off(self):
        result = module(
            train_config=train_config(optimizer=OptimConfig(scheduler="none"))
        ).configure_optimizers()
        self.assertNotIn("lr_scheduler", result)


if __name__ == "__main__":
    unittest.main()


class TestLrProxy(unittest.TestCase):
    """The flat `lr` attribute Lightning's LR finder reads and writes."""

    def test_it_reads_the_optimizer_lr(self):
        model = module()
        self.assertEqual(model.lr, model.train_config.optimizer.lr)

    def test_setting_it_updates_the_config(self):
        # The finder writes its suggestion back through this attribute, so a
        # setter that did not reach the config would silently discard the sweep.
        model = module()
        model.lr = 0.0123
        self.assertEqual(model.train_config.optimizer.lr, 0.0123)

    def test_the_optimizer_actually_uses_the_new_value(self):
        # The proxy must not drift from what trains: configure_optimizers reads
        # train_config directly, so this is what proves the write took effect.
        #
        # The scheduler starts each group *below* its peak (warmup divides the
        # initial rate), so the assertion is proportional rather than equal --
        # what matters is that raising `lr` raises the rates the optimizer runs.
        low, high = module(), module()
        low.lr, high.lr = 0.001, 0.01
        # The *head* group is the one `lr` governs; the backbone group is pinned
        # to `backbone_lr` and deliberately does not follow -- see the next test.
        low_head = max(g["lr"] for g in low.configure_optimizers()["optimizer"].param_groups)
        high_head = max(g["lr"] for g in high.configure_optimizers()["optimizer"].param_groups)
        self.assertGreater(high_head, low_head)
        self.assertAlmostEqual(high_head / low_head, 10.0, places=3)

    def test_the_backbone_rate_is_left_alone(self):
        # backbone_lr is deliberately ~10x lower; a finder sweeping the head's
        # rate must not drag the pretrained encoder along with it.
        model = module()
        before = model.train_config.optimizer.backbone_lr
        model.lr = 0.0123
        self.assertEqual(model.train_config.optimizer.backbone_lr, before)


class TestReleaseValidationState(unittest.TestCase):
    """Validation state must be freeable once the threshold has been fitted.

    An eval-only run validates (to fit the event operating point) and then
    tests. `valid_sample_ids` is cleared only in `on_validation_epoch_start`,
    which never fires again in that flow, so without an explicit release the
    whole validation split stays resident through the test pass -- measured at
    4.75 GB of accelerator memory already held at test batch 0.
    """

    def module(self) -> BlinkLightningModule:
        return BlinkLightningModule(
            model_config=ModelConfig(
                family="lint",
                backbone_pretrained=False,
                backbone_output_dim=4,
                d_model=8,
                num_heads=2,
                cmt_num_layers=1,
                head_hidden_dim=4,
            ),
            train_config=TrainConfig(task="eye_state", loss="bce", max_epochs=1),
            image_size=32,
        )

    def populate_validation(self, model: BlinkLightningModule) -> None:
        for name in model.target_names:
            values = torch.rand(4, 3)
            model.valid_metrics.update(
                name, values, (values > 0.5).float(), torch.ones(4, 3, dtype=torch.bool)
            )
        model.valid_sample_ids = [f"rec|{i:06d}|left" for i in range(4)]

    def test_it_clears_the_sample_ids(self):
        model = self.module()
        self.populate_validation(model)
        model.release_validation_state()
        self.assertEqual(model.valid_sample_ids, [])

    def test_it_clears_the_accumulator(self):
        model = self.module()
        self.populate_validation(model)
        self.assertGreater(model.valid_metrics[model.target_names[0]].valid_count, 0)
        model.release_validation_state()
        self.assertEqual(model.valid_metrics[model.target_names[0]].valid_count, 0)

    def test_it_leaves_the_test_state_alone(self):
        # The release runs between validate() and test(); touching test state
        # would empty the benchmark it is meant to protect.
        model = self.module()
        self.populate_validation(model)
        model.test_sample_ids = ["rec|000000|left"]
        model.release_validation_state()
        self.assertEqual(model.test_sample_ids, ["rec|000000|left"])

    def test_releasing_twice_is_harmless(self):
        model = self.module()
        self.populate_validation(model)
        model.release_validation_state()
        model.release_validation_state()
        self.assertEqual(model.valid_sample_ids, [])

    def test_it_can_run_on_an_untouched_module(self):
        # An eval-only run with no validation data must not crash here.
        model = self.module()
        model.release_validation_state()
        self.assertEqual(model.valid_sample_ids, [])


class TestEvalOnlyTargetIsOptional(unittest.TestCase):
    """A corpus that does not annotate an eval-only target must still score.

    Eval-only targets are scored opportunistically wherever the annotation
    exists. A still-image corpus cannot witness a blink at all, so a batch drawn
    only from CEW or MRL-Eye carries no `blink_presence` key -- and that is
    ordinary, not an error. The trained targets keep the strict lookup, where a
    missing annotation really is a data error.
    """

    def module(self) -> BlinkLightningModule:
        return BlinkLightningModule(
            model_config=ModelConfig(
                family="lint",
                backbone_pretrained=False,
                backbone_output_dim=4,
                d_model=8,
                num_heads=2,
                cmt_num_layers=1,
                head_hidden_dim=4,
            ),
            train_config=TrainConfig(
                task="eye_state",
                loss="bce",
                max_epochs=1,
                eval_targets=[BLINK_PRESENCE],
            ),
            image_size=32,
        )

    def batch(self, with_blink: bool) -> dict:
        size, steps = 2, 1
        data = {
            EYE_IMAGE: torch.rand(size, steps, 3, 32, 32),
            f"{EYE_IMAGE}_mask": torch.ones(size, steps, dtype=torch.bool),
            EYE_STATE: torch.zeros(size, steps),
            f"{EYE_STATE}_mask": torch.ones(size, steps, dtype=torch.bool),
            SAMPLE_KEY: [f"cew|{i:06d}|left" for i in range(size)],
            SOURCE_KEY: ["cew"] * size,
        }
        if with_blink:
            data[BLINK_PRESENCE] = torch.zeros(size, steps)
            data[f"{BLINK_PRESENCE}_mask"] = torch.ones(size, steps, dtype=torch.bool)
        return data

    def test_a_stills_only_batch_does_not_raise(self):
        model = self.module()
        model._shared_step(self.batch(with_blink=False), model.test_metrics)

    def test_the_trained_target_is_still_scored(self):
        model = self.module()
        model._shared_step(self.batch(with_blink=False), model.test_metrics)
        self.assertGreater(model.test_metrics[EYE_STATE].valid_count, 0)

    def test_the_eval_target_stays_empty_when_unannotated(self):
        model = self.module()
        model._shared_step(self.batch(with_blink=False), model.test_metrics)
        self.assertEqual(model.test_metrics[BLINK_PRESENCE].valid_count, 0)

    def test_it_is_scored_where_the_annotation_exists(self):
        model = self.module()
        model._shared_step(self.batch(with_blink=True), model.test_metrics)
        self.assertGreater(model.test_metrics[BLINK_PRESENCE].valid_count, 0)

    def test_a_missing_trained_target_still_raises(self):
        # The strict lookup must survive for supervision the run depends on.
        model = self.module()
        batch = self.batch(with_blink=True)
        del batch[EYE_STATE]
        with self.assertRaises(BatchError):
            model._shared_step(batch, model.test_metrics)


class TestOcclusionReachesTheLoss(unittest.TestCase):
    """The occlusion rule must mask the eye everywhere, not just in a report.

    Applying it in `forward` rather than in a callback is what makes that true:
    a masked eye reaches neither the loss nor the metrics, so the model is never
    asked to predict a lid it cannot see.
    """

    def module(self, occlusion_yaw: float | None) -> BlinkLightningModule:
        return BlinkLightningModule(
            model_config=ModelConfig(
                family="lint",
                backbone_pretrained=False,
                backbone_output_dim=4,
                d_model=8,
                num_heads=2,
                cmt_num_layers=1,
                head_hidden_dim=4,
            ),
            train_config=TrainConfig(
                task="eye_state", loss="bce", max_epochs=1, occlusion_yaw=occlusion_yaw
            ),
            image_size=32,
        )

    def batch(self, yaw: float) -> dict:
        size, steps = 2, 2
        pose = torch.zeros(size, steps, 3)
        pose[..., 0] = yaw
        return {
            EYE_IMAGE: torch.rand(size, steps, 3, 32, 32),
            f"{EYE_IMAGE}_mask": torch.ones(size, steps, dtype=torch.bool),
            EYE_STATE: torch.zeros(size, steps),
            f"{EYE_STATE}_mask": torch.ones(size, steps, dtype=torch.bool),
            HEAD_POSE: pose,
            # The side comes from the key: OmniLoader forwards no `eye_side`.
            SAMPLE_KEY: [f"rec|{i:06d}|{side}" for i, side in enumerate((LEFT, RIGHT))],
            SOURCE_KEY: ["rn30"] * size,
        }

    def test_a_turned_head_masks_one_eye_from_the_metrics(self):
        # With the rule on, the far eye contributes no supervised position.
        model = self.module(occlusion_yaw=45.0)
        model._shared_step(self.batch(yaw=80.0), model.test_metrics)
        counted = model.test_metrics[EYE_STATE].valid_count
        self.assertLess(counted, 4)

    def test_disabling_the_rule_keeps_every_position(self):
        model = self.module(occlusion_yaw=None)
        model._shared_step(self.batch(yaw=80.0), model.test_metrics)
        self.assertEqual(model.test_metrics[EYE_STATE].valid_count, 4)

    def test_a_frontal_head_is_unaffected(self):
        model = self.module(occlusion_yaw=45.0)
        batch = self.batch(yaw=0.0)
        model._shared_step(batch, model.test_metrics)
        self.assertEqual(model.test_metrics[EYE_STATE].valid_count, 4)
