import numpy as np
from layer import Layer

n_embd = 32
block_size = 64


class Head(Layer):
    """one head of self-attention"""

    def __init__(self, input_size, head_size):
        super().__init__()
        # Use much smaller initialization to prevent overflow
        scale = 0.005  # Reduce initialization scale further
        self.key = np.random.randn(input_size, head_size) * scale
        self.query = np.random.randn(input_size, head_size) * scale
        self.value = np.random.randn(input_size, head_size) * scale
        self.tril = np.tril(np.ones((block_size, block_size)))
        self.head_size = head_size

        # Store for backprop
        self.x = None
        self.k = None
        self.q = None
        self.v = None
        self.wei = None

    def forward_propagation(self, input):
        # Handle 2D input (batch, features)
        if len(input.shape) == 2:
            # Reshape to 3D: (batch, 1, features)
            input = input.reshape(input.shape[0], 1, input.shape[1])

        # Now proceed with 3D input: (batch, time-step, channels)
        B, T, C = input.shape

        # Normalize input to prevent overflow
        input_norm = np.linalg.norm(input)
        if input_norm > 10:  # If norm is too large, normalize
            input = input / (input_norm / 10)

        self.x = input  # Save for backprop

        # Calculate k, q, v with safer matrix multiplication
        # Use loop-based approach for better numerical control
        k = np.zeros((B, T, self.head_size))
        q = np.zeros((B, T, self.head_size))
        v = np.zeros((B, T, self.head_size))

        for b in range(B):
            for t in range(T):
                k[b, t] = np.clip(input[b, t] @ self.key, -10, 10)
                q[b, t] = np.clip(input[b, t] @ self.query, -10, 10)
                v[b, t] = np.clip(input[b, t] @ self.value, -10, 10)

        # Save the values for backprop
        self.k, self.q, self.v = k, q, v

        # Fix: compute attention scores with numerical stability
        # Apply scaling before matrix multiplication
        scaling = 1.0 / np.sqrt(self.head_size)
        q_scaled = q * scaling

        # Matrix multiplication with clipping to prevent overflow
        wei = np.zeros((B, T, T))
        for b in range(B):
            wei[b] = np.clip(
                q_scaled[b] @ k[b].T, -10, 10
            )  # Compute per batch with clipping

        # Apply causal mask
        mask = self.tril[:T, :T]
        wei = np.copy(wei)  # Create a copy to avoid modifying the original tensor
        wei[:, mask == 0] = -1e9  # Use finite value instead of -np.inf

        # Custom softmax that's more numerically stable
        # For each row, subtract the maximum value for stability
        wei_max = np.max(wei, axis=2, keepdims=True)
        wei_exp = np.exp(wei - wei_max)
        # Apply mask to set masked positions to zero in exponential space
        mask_expanded = np.expand_dims(mask, 0)
        mask_expanded = np.repeat(mask_expanded, B, axis=0)
        wei_exp = wei_exp * mask_expanded
        wei_sum = np.sum(wei_exp, axis=2, keepdims=True)
        wei = wei_exp / (
            wei_sum + 1e-10
        )  # Add small epsilon to prevent division by zero

        self.wei = wei  # Save for backprop

        # Perform the weighted aggregation of the values
        out = np.zeros_like(v)
        for b in range(B):
            out[b] = wei[b] @ v[b]  # (T, T) @ (T, hs) -> (T, hs)

        return out

    def backward_propagation(self, output_error, learning_rate):
        # output_error shape: (B, T, hs)
        B, T, _ = output_error.shape

        # Clip output_error to prevent overflow
        output_error = np.clip(output_error, -10, 10)

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
        dv = np.zeros_like(self.v)
        for b in range(B):
            dv[b] = np.clip(self.wei[b].T @ output_error[b], -10, 10)  # With clipping

        # Gradient w.r.t. wei (before softmax)
        dwei = np.zeros((B, T, T))
        for b in range(B):
            dwei[b] = np.clip(output_error[b] @ self.v[b].T, -10, 10)  # With clipping

        # Apply mask to dwei (same as forward pass)
        mask = self.tril[:T, :T]
        dwei[:, mask == 0] = 0  # Zero out gradients for masked positions

        # Gradient w.r.t. q
        scaling = self.head_size**-0.5
        dk_transpose = dwei * scaling  # (B, T, T)
        dq = np.zeros_like(self.q)
        for b in range(B):
            dq[b] = np.clip(dk_transpose[b] @ self.k[b], -10, 10)  # With clipping

        # Gradient w.r.t. k
        dk = np.zeros_like(self.k)
        for b in range(B):
            dk[b] = np.clip(dwei[b].T @ self.q[b], -10, 10)  # With clipping

        # Weight errors - simplified approach with clipping
        value_error = np.zeros_like(self.value)
        query_error = np.zeros_like(self.query)
        key_error = np.zeros_like(self.key)

        # Calculate weight errors for each sample in batch and sum them
        for b in range(B):
            value_error += np.clip(self.x[b].T @ dv[b], -10, 10)
            query_error += np.clip(self.x[b].T @ dq[b], -10, 10)
            key_error += np.clip(self.x[b].T @ dk[b], -10, 10)

        # Gradient w.r.t. input x: dE/dX
        input_error = np.zeros_like(self.x)
        for b in range(B):
            input_error[b] = np.clip(
                dv[b] @ self.value.T + dq[b] @ self.query.T + dk[b] @ self.key.T,
                -10,
                10,
            )

        # Update parameters (similar to FC layer update)
        # Use much smaller learning rate for attention weights
        effective_lr = learning_rate * 0.01
        self.key -= effective_lr * key_error
        self.query -= effective_lr * query_error
        self.value -= effective_lr * value_error

        return input_error


class AttentionLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        super().__init__()
        head_size = 16  # Use smaller head size

        self.head = Head(input_size, head_size)

        # Linear projection with small weights
        scale = 0.005
        self.proj_weights = np.random.randn(head_size, output_size) * scale
        self.proj_bias = np.zeros(output_size)

        # Store for backprop
        self.input = None
        self.head_output = None

    def forward_propagation(self, input):
        self.input = input  # Save for backprop

        # Process through the single attention head
        out = self.head.forward_propagation(input)
        self.head_output = out  # Save for backprop

        # Project to output_size dimension
        if len(out.shape) == 3:
            # If 3D (batch, time-step, head_size)
            # Reshape to 2D for the output layer
            B, T, _ = out.shape
            if T == 1:  # If only one time step
                out = out.reshape(B, -1)  # (B, head_size)
                result = np.clip(out @ self.proj_weights + self.proj_bias, -10, 10)
                return result
            else:
                # Apply projection for each time step
                result = np.zeros((B, T, self.proj_weights.shape[1]))
                for b in range(B):
                    for t in range(T):
                        result[b, t] = np.clip(
                            out[b, t] @ self.proj_weights + self.proj_bias, -10, 10
                        )
                return result
        else:
            # If already 2D
            return np.clip(out @ self.proj_weights + self.proj_bias, -10, 10)

    def backward_propagation(self, output_error, learning_rate):
        # Handle 2D output_error (batch, output_size)
        if len(output_error.shape) == 2:
            # Reshape to (batch, 1, output_size)
            output_error = output_error.reshape(
                output_error.shape[0], 1, output_error.shape[1]
            )

        # Clip output_error to prevent overflow
        output_error = np.clip(output_error, -10, 10)

        B, T, _ = output_error.shape

        # Gradient for projection weights: dE/dW
        weights_error = np.zeros_like(self.proj_weights)

        if self.head_output is None:
            print("Error: head_output is None. Did you run forward() first?")
            return np.zeros_like(self.input)

        # Calculate weight errors for each sample in batch and sum them
        for b in range(B):
            for t in range(T):
                # Use outer product with clipping
                outer = np.outer(self.head_output[b, t], output_error[b, t])
                weights_error += np.clip(outer, -10, 10)

        # Gradient for bias: dE/dB
        bias_error = np.sum(output_error, axis=(0, 1))

        # Gradient w.r.t. head output: dE/dY_head
        head_error = np.zeros_like(self.head_output)
        for b in range(B):
            for t in range(T):
                head_error[b, t] = np.clip(
                    output_error[b, t] @ self.proj_weights.T, -10, 10
                )

        # Backpropagate through head
        input_error = self.head.backward_propagation(head_error, learning_rate)

        # If input was 2D, ensure output is also 2D
        if len(self.input.shape) == 2:
            input_error = input_error.reshape(input_error.shape[0], -1)

        # Use much smaller learning rate for attention projection
        effective_lr = learning_rate * 0.01
        self.proj_weights -= effective_lr * weights_error
        self.proj_bias -= effective_lr * bias_error

        return input_error
