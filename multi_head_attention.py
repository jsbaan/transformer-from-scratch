import math
import unittest
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """
    Cheated by looking at
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html.
    To allow for arbitrary inputs, an additional param input_dim could be passed.
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
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
        assert hidden_dim % num_heads == 0
        self.qkv_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.qkv_proj = nn.Linear(hidden_dim, 3 * num_heads * self.qkv_dim)
        self.o_proj = nn.Linear(num_heads * self.qkv_dim, hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        """ Weight initialization taken from the UvA DL1 PyTorch Transformer tutorial. """
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(
        self,
        x: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Perform multi-head attention using one projection matrix.
        Self attention is performed when encoder_hidden_states is None.
        x is (n_batches, sequence_length, hidden_dim)

        Cross-attention is performed when encoder_hidden_states are not None.
        In that case, input x represents the decoder hidden states with dim=(n_batches, tgt_sequence_length, hidden_dim)
        and encoder_hidden_state has dim=(n_batches, src_sequence_length, hidden_dim)
        """
        n_batches, sequence_length, hidden_dim = x.size()

        if encoder_hidden_states is None:
            q, v, k = self._self_attention_projection(x)
        else:
            q, v, k = self._cross_attention_projection(encoder_hidden_states, x)

        # Swap dimensions to (n_batches, n_heads, seq_len, qkv_dim). This is required for scaled dot product.
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute (contextualized) value vector for each "head"
        values, attn = self.scaled_dot_product(q, k, v, mask=mask)

        # Concatenate contextualized value vectors from all heads
        values = values.reshape(n_batches, sequence_length, hidden_dim)

        # Linearly transform the concatenation of all heads' value vectors (8*64=512) to the original hidden dim (512)
        output = self.o_proj(values)
        return output

    def _self_attention_projection(self, x: torch.FloatTensor):
        """
        Project x and interpret the result as chunks that represent q, k and v vectors for every head.
        x can be encoder or decoder hidden states, depending on which one calls this MHA module.

        """
        n_batches, sequence_length, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(n_batches, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _cross_attention_projection(
        self,
        encoder_hidden_states: torch.FloatTensor,
        decoder_hidden_states: torch.FloatTensor,
    ):
        """
        Project the decoder hidden states into query vectors and the encoder hidden states into key and value vectors.
        The columns of W_proj determine how much independent linear combinations of the input we obtain - which we
        then interpret as heads and qkv vectors. Thus we can simply split the weight matrix and project the decoder
        hidden states x into q separately from projecting the encoder_hidden_states into k and v.
        """
        n_batches, src_sequence_length, hidden_dim = encoder_hidden_states.shape
        n_batches, tgt_sequence_length, hidden_dim = decoder_hidden_states.shape

        # Split weight matrix and bias
        w_q, w_kv = self.qkv_proj.weight.split([hidden_dim, 2 * hidden_dim])
        b_q, b_kv = (
            self.qkv_proj.bias.split([hidden_dim, 2 * hidden_dim])
            if self.qkv_proj.bias is not None
            else (None, None)
        )

        # Project encoder_hidden_states into k's, and v's
        k, v = (
            F.linear(input=encoder_hidden_states, weight=w_kv, bias=b_kv)
            .reshape(n_batches, src_sequence_length, self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )

        # Project decoder hidden states into q's
        q = F.linear(input=decoder_hidden_states, weight=w_q, bias=b_q).reshape(
            n_batches, tgt_sequence_length, self.num_heads, self.qkv_dim
        )

        return q, k, v

    @staticmethod
    def scaled_dot_product(
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        The query, key and value tensors must have dimensionality (n_batches, num_heads, sequence_length, qkv_dim).
        For cross-attention, the sequence length may differ as q is projected from decoder hidden states and kv aren't
        """

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
        attention = F.softmax(attn_logits, dim=-1)

        # Weighted sum of value vectors for each input token using attention scores -> new contextualized representation
        # (n_batches, num_heads, sequence_length, qkv_dim)
        values = torch.matmul(attention, v)
        return values, attention


class TestMultiHeadAttention(unittest.TestCase):
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
        torch.testing.assert_close(torch.sum(attention_scores, dim=-1), expected)

        self.assertEqual(torch.any(torch.isnan(values)), False)
        self.assertEqual(True in torch.isnan(attention_scores), False)

    def test_scaled_dot_product_mask(self):
        mha = MultiHeadAttention(hidden_dim=512, num_heads=8)
        q = torch.randn(2, 8, 10, 512, dtype=torch.float)
        k = torch.randn(2, 8, 10, 512, dtype=torch.float)
        v = torch.randn(2, 8, 10, 512, dtype=torch.float)
        mask = torch.BoolTensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )

        _, attention_scores = mha.scaled_dot_product(q, k, v, mask=mask)
        self.assertEqual(torch.any(torch.isnan(attention_scores)), False)

        # For the first sequence we expect the last two (8-10) attention scores for every attention distribution
        # for every head to be exactly zero due to the mask we defined above. The rest should be strictly non-zero.
        self.assertEqual(torch.all(attention_scores[0, :, :, 8:] == 0), True)
        self.assertEqual(torch.any(attention_scores[0, :, :, :8] == 0), False)

        # Each attention distribution should sum up to one (all values after summing should be 1)
        expected = torch.FloatTensor([1.0]).repeat((2, 8, 10))
        torch.testing.assert_close(torch.sum(attention_scores, dim=-1), expected)

        # For the second sequence in the batch all attention scores should be nonzero because the mask is all ones
        self.assertEqual(torch.any(attention_scores[1] == 0), False)

    def test_mha_self_attention_forward(self):
        mha = MultiHeadAttention(512, 8)
        x = torch.randn(4, 10, 512, dtype=torch.float)
        output = mha.forward(x)
        self.assertEqual(output.shape, (4, 10, 512))
        self.assertEqual(torch.any(torch.isnan(output)), False)

    def test_mha_cross_attention_forward(self):
        mha = MultiHeadAttention(512, 8)
        decoder_hidden_states = torch.randn(4, 2, 512, dtype=torch.float)
        encoder_hidden_states = torch.randn(4, 10, 512, dtype=torch.float)
        output = mha.forward(
            x=decoder_hidden_states, encoder_hidden_states=encoder_hidden_states
        )
        self.assertEqual(output.shape, (4, 2, 512))
        self.assertEqual(torch.any(torch.isnan(output)), False)


if __name__ == "__main__":
    unittest.main()
