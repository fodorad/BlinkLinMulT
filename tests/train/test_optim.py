"""Tests for optimizer and schedule construction."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from blinklinmult.train.config import OptimConfig
from blinklinmult.train.optim import (
    BACKBONE_GROUP,
    HEAD_GROUP,
    MIN_ONECYCLE_STEPS,
    OptimError,
    build_optimizer,
    build_scheduler,
    parameter_groups,
)


class FakeEncoder(nn.Module):
    """Stands in for the eye encoder: a backbone plus a projection."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.project = nn.Linear(8, 4)


class FakeModel(nn.Module):
    """A model shaped like the real one: encoder.backbone plus a sequence model."""

    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.sequence_model = nn.Linear(4, 2)

    def forward(self, x):
        return self.sequence_model(x)


class TestParameterGroups(unittest.TestCase):
    def test_two_groups_when_backbone_lr_is_set(self):
        groups = parameter_groups(FakeModel(), OptimConfig(backbone_lr=1e-4))
        self.assertEqual([g["name"] for g in groups], [BACKBONE_GROUP, HEAD_GROUP])

    def test_the_groups_carry_their_own_rates(self):
        groups = parameter_groups(FakeModel(), OptimConfig(lr=1e-3, backbone_lr=1e-5))
        rates = {g["name"]: g["lr"] for g in groups}
        self.assertEqual(rates[BACKBONE_GROUP], 1e-5)
        self.assertEqual(rates[HEAD_GROUP], 1e-3)

    def test_one_group_when_backbone_lr_is_none(self):
        groups = parameter_groups(FakeModel(), OptimConfig(backbone_lr=None))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["name"], HEAD_GROUP)

    def test_every_trainable_parameter_lands_in_exactly_one_group(self):
        model = FakeModel()
        groups = parameter_groups(model, OptimConfig(backbone_lr=1e-4))
        grouped = [p for g in groups for p in g["params"]]
        self.assertEqual(len(grouped), len(list(model.parameters())))
        # Identity, not equality: a parameter in two groups would be stepped twice.
        self.assertEqual(len({id(p) for p in grouped}), len(grouped))

    def test_only_backbone_parameters_are_in_the_backbone_group(self):
        model = FakeModel()
        groups = parameter_groups(model, OptimConfig(backbone_lr=1e-4))
        backbone = next(g for g in groups if g["name"] == BACKBONE_GROUP)
        expected = {id(p) for p in model.encoder.backbone.parameters()}
        self.assertEqual({id(p) for p in backbone["params"]}, expected)

    def test_a_frozen_backbone_collapses_to_one_group(self):
        model = FakeModel()
        for parameter in model.encoder.backbone.parameters():
            parameter.requires_grad = False

        groups = parameter_groups(model, OptimConfig(backbone_lr=1e-4))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["name"], HEAD_GROUP)

    def test_frozen_parameters_are_excluded(self):
        model = FakeModel()
        model.sequence_model.weight.requires_grad = False
        groups = parameter_groups(model, OptimConfig(backbone_lr=None))
        grouped = {id(p) for p in groups[0]["params"]}
        self.assertNotIn(id(model.sequence_model.weight), grouped)


class TestBuildOptimizer(unittest.TestCase):
    def test_every_supported_optimizer_builds(self):
        for name in ("adam", "adamw", "radam", "sgd"):
            with self.subTest(optimizer=name):
                optimizer = build_optimizer(OptimConfig(name=name), FakeModel())
                self.assertIsInstance(optimizer, torch.optim.Optimizer)

    def test_group_rates_survive_construction(self):
        optimizer = build_optimizer(OptimConfig(lr=1e-3, backbone_lr=1e-5), FakeModel())
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-5, 1e-3])

    def test_weight_decay_is_applied(self):
        optimizer = build_optimizer(OptimConfig(weight_decay=0.05), FakeModel())
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.05)

    def test_sgd_receives_its_momentum(self):
        optimizer = build_optimizer(OptimConfig(name="sgd", momentum=0.95), FakeModel())
        self.assertEqual(optimizer.param_groups[0]["momentum"], 0.95)

    def test_a_fully_frozen_model_raises(self):
        model = FakeModel()
        for parameter in model.parameters():
            parameter.requires_grad = False
        with self.assertRaises(OptimError) as ctx:
            build_optimizer(OptimConfig(), model)
        self.assertIn("backbone_freeze", str(ctx.exception))

    def test_a_step_actually_moves_the_parameters(self):
        model = FakeModel()
        optimizer = build_optimizer(OptimConfig(lr=0.1), model)
        before = model.sequence_model.weight.detach().clone()

        model(torch.rand(2, 4)).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.sequence_model.weight))


class TestBuildScheduler(unittest.TestCase):
    def optimizer(self, **kwargs) -> torch.optim.Optimizer:
        return build_optimizer(OptimConfig(**kwargs), FakeModel())

    def test_none_returns_no_schedule(self):
        self.assertIsNone(build_scheduler(OptimConfig(scheduler="none"), self.optimizer(), 10, 5))

    def test_onecycle_steps_per_batch(self):
        result = build_scheduler(OptimConfig(scheduler="onecycle"), self.optimizer(), 10, 5)
        self.assertEqual(result["interval"], "step")
        self.assertIsInstance(result["scheduler"], torch.optim.lr_scheduler.OneCycleLR)

    def test_onecycle_preserves_per_group_peaks(self):
        # A single max_lr would flatten the backbone's lower rate.
        optimizer = self.optimizer(lr=1e-3, backbone_lr=1e-5)
        build_scheduler(OptimConfig(scheduler="onecycle"), optimizer, 100, 5)
        peaks = [g["max_lr"] for g in optimizer.param_groups]
        self.assertEqual(peaks, [1e-5, 1e-3])

    def test_onecycle_falls_back_on_a_degenerate_run(self):
        # Fewer steps than a warmup/peak/anneal needs: a constant rate rather
        # than a ZeroDivisionError inside the scheduler.
        self.assertIsNone(
            build_scheduler(OptimConfig(scheduler="onecycle"), self.optimizer(), 1, 1)
        )

    def test_onecycle_builds_at_the_minimum_step_count(self):
        result = build_scheduler(
            OptimConfig(scheduler="onecycle"), self.optimizer(), MIN_ONECYCLE_STEPS, 1
        )
        self.assertIsNotNone(result)

    def test_onecycle_survives_the_degenerate_pct_start(self):
        # warmup_ratio 0.1 over exactly 10 steps puts the phase boundary on step
        # 0, which OneCycleLR divides by. A plausible smoke configuration.
        result = build_scheduler(
            OptimConfig(scheduler="onecycle", warmup_ratio=0.1),
            self.optimizer(),
            steps_per_epoch=10,
            max_epochs=1,
        )
        self.assertIsNotNone(result)

    def test_onecycle_runs_to_completion(self):
        optimizer = self.optimizer()
        result = build_scheduler(OptimConfig(scheduler="onecycle"), optimizer, 4, 3)
        scheduler = result["scheduler"]
        for _ in range(12):
            optimizer.step()
            scheduler.step()

    def test_cosine_is_step_wise(self):
        result = build_scheduler(OptimConfig(scheduler="cosine"), self.optimizer(), 10, 5)
        self.assertEqual(result["interval"], "step")

    def test_cosine_with_warmup_does_not_end_at_zero(self):
        # The cosine's period must cover only the post-warmup steps; given the
        # full run it completes early and sits at lr=0 for the tail.
        optimizer = self.optimizer(lr=1e-3)
        result = build_scheduler(
            OptimConfig(scheduler="cosine", warmup_ratio=0.2), optimizer, 10, 5
        )
        scheduler = result["scheduler"]
        for _ in range(50):
            optimizer.step()
            scheduler.step()
        self.assertGreater(optimizer.param_groups[-1]["lr"], 0.0)

    def test_cosine_without_warmup_has_no_chained_schedule(self):
        result = build_scheduler(
            OptimConfig(scheduler="cosine", warmup_ratio=0.0), self.optimizer(), 10, 5
        )
        self.assertIsInstance(result["scheduler"], torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_plateau_is_epoch_wise_and_monitors_the_primary_metric(self):
        result = build_scheduler(OptimConfig(scheduler="plateau"), self.optimizer(), 10, 20)
        self.assertEqual(result["interval"], "epoch")
        self.assertEqual(result["monitor"], "valid/mean_f1")

    def test_unknown_scheduler_raises(self):
        config = OptimConfig()
        object.__setattr__(config, "scheduler", "exponential")
        with self.assertRaises(OptimError):
            build_scheduler(config, self.optimizer(), 10, 5)


if __name__ == "__main__":
    unittest.main()
