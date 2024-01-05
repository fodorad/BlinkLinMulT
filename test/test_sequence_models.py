import unittest
import torch
from blinklinmult.models.BlinkLinMulT import DenseNet121, BlinkLinT, BlinkLinMulT


class TestSequenceModels(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size: int = 16
        self.time_dim: int = 15

    def test_densenet121(self):
        model = DenseNet121(output_dim=1, weights='densenet121-union')
        x = torch.zeros((self.batch_size, 3, 64, 64)) # (B, C, H, W)
        y_pred = model(x)
        self.assertEqual(y_pred.shape, (self.batch_size, 1))

    def test_densenetlint(self):
        model = BlinkLinT(output_dim=1, weights='blinklint-union')
        x = torch.zeros((self.batch_size, self.time_dim, 3, 64, 64)) # (B, L, C, H, W)
        y_seq = model(x)
        self.assertEqual(y_seq.shape, (self.batch_size, self.time_dim, 1))

    def test_blinklinmult(self):
        model = BlinkLinMulT(input_dim=160, output_dim=1, weights='blinklinmult-union')
        seq_64 = torch.zeros((self.batch_size, self.time_dim, 3, 64, 64)) # (B, L, C, H, W)
        seq_160 = torch.zeros((self.batch_size, self.time_dim, 160)) # (B, L, C)
        y_cls, y_seq = model([seq_64, seq_160])
        self.assertEqual(y_cls.shape, (self.batch_size, 1))
        self.assertEqual(y_seq.shape, (self.batch_size, self.time_dim, 1))


if __name__ == '__main__':
    unittest.main()