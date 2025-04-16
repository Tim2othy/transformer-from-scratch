import numpy as np
from layer import Layer
from layers import softmax

n_embd = 32
block_size = 64


class Head(Layer):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = np.random.rand(n_embd, head_size) * 0.1 - 0.05
        self.query = np.random.rand(n_embd, head_size) * 0.1 - 0.05
        self.value = np.random.rand(n_embd, head_size) * 0.1 - 0.05
        self.tril = np.tril(np.ones((block_size, block_size)))

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = x @ self.key  # (B,T,hs)
        q = x @ self.query  # (B,T,hs)
        v = x @ self.value  # (B,T,hs)

        # compute attention scores ("affinities")
        wei = (
            q @ np.transpose(k, (0, 2, 1)) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # Apply causal mask
        mask = self.tril[:T, :T]
        wei[:, mask == 0] = -np.inf  # Mask future positions
        wei = softmax(wei)
        # perform the weighted aggregation of the values

        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class AttentionLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        super().__init__()
        # Use head_size = input_size
        self.head = Head(input_size)  # Just one head

        # Linear projection from head output (input_size) to output_size
        self.proj_weights = np.random.rand(input_size, output_size) * 0.1 - 0.05
        self.proj_bias = np.zeros(output_size)

    def forward(self, x):
        # Process through the single attention head
        out = self.head.forward(x)
        # Project to output_size dimension
        out = out @ self.proj_weights + self.proj_bias
        return out
