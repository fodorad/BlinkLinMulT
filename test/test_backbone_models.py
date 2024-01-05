import unittest
import torch
from blinklinmult.models.backbone import Dense, EyeNet, ResNet50


class TestBackboneModels(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size: int = 16

    def test_dense(self):
        model = Dense(input_dim=100, output_dim=1)
        x = torch.zeros((self.batch_size, 100))
        y_pred = model(x)
        self.assertEqual(y_pred.shape, (self.batch_size, 1))

    def test_eyenet(self):
        model = EyeNet(output_dim=1)
        x = torch.zeros((self.batch_size, 3, 64, 64))
        y_pred = model(x)
        self.assertEqual(y_pred.shape, (self.batch_size, 1))

    def test_eyenet(self):
        model = ResNet50(output_dim=1)
        x = torch.zeros((self.batch_size, 3, 64, 64))
        y_pred = model(x)
        self.assertEqual(y_pred.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()