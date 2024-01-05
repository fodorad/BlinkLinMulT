import torch
from torch import nn


class GRU(nn.Module):

    def __init__(self, input_dim,
                       hidden_dim: int = 256,
                       layer_dim: int = 3,
                       dropout_prob: float = 0.2,
                       return_sequences: bool = True):
        super().__init__()
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.return_sequences = return_sequences
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        h0 = h0.to(x.device)
        self.gru = self.gru.to(x.device)

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0)

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        if not self.return_sequences:
            out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        # out = self.fc(out)

        return out