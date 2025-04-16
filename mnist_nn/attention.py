import numpy as np
from layer import Layer
from layers import softmax

n_embd = 32
block_size = 64


class Head(Layer):
    """one head of self-attention"""

    def __init__(self, input_size, head_size):
        super().__init__()
        # Now properly using input_size instead of n_embd
        self.key = np.random.rand(input_size, head_size) * 0.1 - 0.05
        self.query = np.random.rand(input_size, head_size) * 0.1 - 0.05
        self.value = np.random.rand(input_size, head_size) * 0.1 - 0.05
        self.tril = np.tril(np.ones((block_size, block_size)))
        self.head_size = head_size

        # Store for backprop - initialize as empty arrays
        self.x = None
        self.k = None
        self.q = None
        self.v = None
        self.wei = None

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        self.x = x  # Save for backprop

        # Calculate k, q, v
        k = x @ self.key  # (B,T,hs)
        q = x @ self.query  # (B,T,hs)
        v = x @ self.value  # (B,T,hs)

        # Fix: Actually save the values (missing =)
        self.k, self.q, self.v = k, q, v  # Save for backprop

        # compute attention scores ("affinities")
        wei = (
            q @ np.transpose(k, (0, 2, 1)) * self.head_size**-0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # Apply causal mask
        mask = self.tril[:T, :T]
        wei = np.copy(wei)  # Create a copy to avoid modifying the original tensor
        wei[:, mask == 0] = -np.inf  # Mask future positions

        # Softmax across the last dimension
        wei = softmax(wei)  # Apply softmax across the last dimension
        self.wei = wei  # Save for backprop

        # perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

    def backward_propagation(self, output_error, learning_rate):
        # output_error shape: (B, T, hs)
        B, T, _ = output_error.shape

        # Fix: Check if self.wei, self.v, etc. are None before using them
        if (
            self.wei is None
            or self.v is None
            or self.q is None
            or self.k is None
            or self.x is None
        ):
            print(
                "Error: One of the saved values is None. Did you run forward() first?"
            )
            return np.zeros_like(self.x) if self.x is not None else np.zeros((B, T, T))

        # Gradient w.r.t. v: dE/dV
        # Fix: The shapes don't match for matrix multiplication
        # Use a loop for simplicity and reliability
        dv = np.zeros_like(self.v)
        for b in range(B):
            dv[b] = self.wei[b].T @ output_error[b]  # (T, T) @ (T, hs) -> (T, hs)

        # Gradient w.r.t. wei (before softmax)
        # Fix: The shapes don't match for matrix multiplication
        dwei = np.zeros((B, T, T))
        for b in range(B):
            dwei[b] = output_error[b] @ self.v[b].T  # (T, hs) @ (hs, T) -> (T, T)

        # Apply mask to dwei (same as forward pass)
        mask = self.tril[:T, :T]
        dwei[:, mask == 0] = 0  # Zero out gradients for masked positions

        # Gradient w.r.t. q
        scaling = self.head_size**-0.5
        dk_transpose = dwei * scaling  # (B, T, T)
        dq = np.zeros_like(self.q)
        for b in range(B):
            dq[b] = dk_transpose[b] @ self.k[b]  # (T, T) @ (T, hs) -> (T, hs)

        # Gradient w.r.t. k
        dk = np.zeros_like(self.k)
        for b in range(B):
            dk[b] = dwei[b].T @ self.q[b]  # (T, T) @ (T, hs) -> (T, hs)

        # Weight errors - simplified approach
        value_error = np.zeros_like(self.value)
        query_error = np.zeros_like(self.query)
        key_error = np.zeros_like(self.key)

        # Calculate weight errors for each sample in batch and sum them
        for b in range(B):
            value_error += self.x[b].T @ dv[b]
            query_error += self.x[b].T @ dq[b]
            key_error += self.x[b].T @ dk[b]

        # Gradient w.r.t. input x: dE/dX
        input_error = np.zeros_like(self.x)
        for b in range(B):
            input_error[b] = (
                dv[b] @ self.value.T + dq[b] @ self.query.T + dk[b] @ self.key.T
            )

        # Update parameters (similar to FC layer update)
        self.key -= learning_rate * key_error
        self.query -= learning_rate * query_error
        self.value -= learning_rate * value_error

        return input_error


class AttentionLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        super().__init__()
        head_size = 16  # Typical head size, can be adjusted
        self.input_size = input_size
        self.output_size = output_size
        self.head_size = head_size

        # Pass input_size to Head constructor
        self.head = Head(input_size, head_size)

        # Linear projection from head_size to output_size
        self.proj_weights = np.random.rand(head_size, output_size) * 0.1 - 0.05
        self.proj_bias = np.zeros(output_size)

        # Store for backprop
        self.head_output = None

    def forward_propagation(self, input):
        # Process through the single attention head
        out = self.head.forward(input)
        self.head_output = out  # Save for backprop

        # Project to output_size dimension
        out = out @ self.proj_weights + self.proj_bias
        return out

    def backward_propagation(self, output_error, learning_rate):
        # output_error shape: (B, T, output_size)
        B, T, _ = output_error.shape

        # Fix: Check if head_output is None
        if self.head_output is None:
            print("Error: head_output is None. Did you run forward() first?")
            return np.zeros((B, T, self.input_size))  # Return zeros with correct shape

        # Gradient for projection weights: dE/dW
        weights_error = np.zeros_like(self.proj_weights)

        # Calculate weight errors for each sample in batch and sum them
        for b in range(B):
            for t in range(T):
                weights_error += np.outer(self.head_output[b, t], output_error[b, t])

        # Gradient for bias: dE/dB
        bias_error = np.sum(output_error, axis=(0, 1))

        # Gradient w.r.t. head output: dE/dY_head
        head_error = output_error @ self.proj_weights.T

        # Backpropagate through head
        input_error = self.head.backward_propagation(head_error, learning_rate)

        # Update parameters (similar to FC layer update)
        self.proj_weights -= learning_rate * weights_error
        self.proj_bias -= learning_rate * bias_error

        return input_error
