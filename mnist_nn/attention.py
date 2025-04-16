import numpy as np
from layer import Layer
from layers import softmax


class Head(Layer):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = np.random.rand(n_embd, head_size) - 0.5
        self.query = np.random.rand(n_embd, head_size) - 0.5
        self.value = np.random.rand(n_embd, head_size) - 0.5
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = softmax(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
