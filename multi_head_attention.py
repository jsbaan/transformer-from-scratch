import math
import unittest
from typing import Optional

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """
    Cheated by looking at
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html.
    To allow for arbitrary inputs, an additional param input_dim could be passed.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()

        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        """
        Each element in x @ W_proj will be the dot product between a row in x and a columns in W. This means that 
        multiplying input x with a (hidden_dim, 3*hidden_dim) matrix will be identical to performing three matrix 
        multiplications with three separate W matrices of (hidden_dim, hidden_dim). This in turn is identical to 
        performing, 8 separate matrix multiplications of (hidden_dim, hidden_dim/8). In other words; each extra column 
        in will result in another linear combination (projection) of each row (embedding) in x. 

        Example (discarding batch size): d_x = (10, 512) and d_W_proj = (512, 3*512). This results in 
        3 times 512 linear combinations of each row in x. We can further interpret each block of 512 as 8 blocks of 64
        that represent the attention heads.

        Long story short: this projection can be interpreted as performing a different projection q, k and v, and then
        for each of the 8 heads in q, k and v.
        """
        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """ I cannot find this anywhere in the paper. Copied this from the UvA DL tutorial """
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    @staticmethod
    def scaled_dot_product(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """ query, key and value tensors have dimensionality (n_batches, num_heads, sequence_length, qkv_dim) """

        # Compute attention logits: dot product between each query and key vector (through one matrix multiplication)
        # Results in un-normalized attention scores for each position's query vector to each position's key vector
        # k^T dim(n_batches, num_heads, qkv_dim, seq_length), output dim(n_batches, num_heads, seq_length, seq_length)
        attn_logits = torch.matmul(q, torch.transpose(k, -2, -1),)

        # Scale logits by constant to create less spiky softmax distribution
        attn_logits = attn_logits / math.sqrt(q.size()[-1])

        # Apply attention mask (for pad tokens and future-masking in cross-attention)
        if mask is not None:
            attn_logits = (
                attn_logits.permute(1, 0, 2, 3)
                .masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
                .permute(1, 0, 2, 3)
            )

        # Transform logits to attention probability distribution (one distribution per non-masked token index)
        attention = nn.functional.softmax(attn_logits, dim=-1)

        # Weighted sum of value vectors for each input token using attention scores -> new contextualized representation
        # (n_batches, num_heads, sequence_length, qkv_dim)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        """ Perform multi-head attention"""
        n_batches, sequence_length, hidden_dim = x.size()

        # Project input and extract the chunks that we interpret as Q, K and V
        qkv = self.qkv_proj(x)  # todo why can't I simply do .chunk(3, dim=-1) here?
        qkv = qkv.reshape(n_batches, sequence_length, self.num_heads, 3 * self.qkv_dim)

        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute contextualized value vector for each "head": values dim(n_batches, num_heads, seq_len, qkv_dim)
        values, attn = self.scaled_dot_product(q, k, v, mask=mask)

        # Concatenate contextualized value vectors from all heads
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(n_batches, sequence_length, hidden_dim)

        output = self.o_proj(values)
        return output


class TestMultiHeadAttention(unittest.TestCase):
    JB_DISABLE_BUFFERING = 1

    def test_scaled_dot_product(self):
        mha = MultiHeadAttention(512, 8)
        q = torch.randn(4, 8, 10, 512)
        k = torch.randn(4, 8, 10, 512)
        v = torch.randn(4, 8, 10, 512)

        values, attention_scores = mha.scaled_dot_product(q, k, v)

        self.assertEqual(values.shape, (4, 8, 10, 512))
        self.assertEqual(attention_scores.shape, (4, 8, 10, 10))

        # Each attention distribution should sum up to one
        expected = torch.FloatTensor([1.0]).repeat((4, 8, 10))
        self.assertEqual(
            torch.all(torch.isclose(torch.sum(attention_scores, dim=-1), expected)),
            True,
        )

        self.assertEqual(True in torch.isnan(values), False)
        self.assertEqual(True in torch.isnan(attention_scores), False)

    def test_scaled_dot_product_mask(self):
        mha = MultiHeadAttention(hidden_dim=512, num_heads=8)
        q = torch.randn(2, 8, 10, 512)
        k = torch.randn(2, 8, 10, 512)
        v = torch.randn(2, 8, 10, 512)
        mask = torch.BoolTensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )

        _, attention_scores = mha.scaled_dot_product(q, k, v, mask=mask)

        # For the first sequence we expect the last two (8-10) attention scores for every attention distribution
        # for every head to be exactly zero due to the mask we defined above. The rest should be strictly non-zero.
        self.assertEqual(torch.all(attention_scores[0, :, :, 8:] == 0), True)
        self.assertEqual(torch.any(attention_scores[0, :, :, :8] == 0), False)

        # Each attention distribution should sum up to one (all values after summing should be 1)
        expected = torch.FloatTensor([1.0]).repeat((2, 8, 10))
        torch.all(torch.isclose(torch.sum(attention_scores, dim=-1), expected))

        # For the second sequence in the batch all attention scores should be nonzero because the mask is all ones
        self.assertEqual(torch.any(attention_scores[1] == 0), False)

    def test_mha_forward(self):
        mha = MultiHeadAttention(512, 8)
        x = torch.randn(4, 10, 512)
        output = mha.forward(x)
        self.assertEqual(output.shape, (4, 10, 512))
        self.assertEqual(True in torch.isnan(output), False)


if __name__ == "__main__":
    unittest.main()
