import math
import unittest
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from utils import construct_future_mask


class MultiHeadAttention(nn.Module):
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
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Perform multi-head attention using one projection matrix.
        Self attention is performed when encoder_hidden_states is None.
        Otherwise, cross-attention is performed
        In that case, input x represents the decoder hidden states.

        :param x: (batch_size, sequence_length, hidden_dim)
        :param encoder_hidden_states: (batch_size, src_sequence_length, hidden_dim)
        :param src_padding_mask: Dim (batch_size, src_sequence_length).Used for encoder self-attention and
            cross-attention to handle pad tokens. Masks all incoming "connections" or "logits" from any token position
            to any pad token in a sequence.
        :param future_mask: Dim (tgt_sequence_length, tgt_sequence_length). Used for decoder self-attention to avoid
            any token i attending to a token >i, i.e. "peaking".
        :return:
        """
        batch_size, sequence_length, hidden_dim = x.size()

        if encoder_hidden_states is None:
            q, v, k = self._self_attention_projection(x)
        else:
            q, v, k = self._cross_attention_projection(encoder_hidden_states, x)

        # Swap dimensions to (batch_size, n_heads, seq_len, qkv_dim). Required for the matrix multiplication below
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Compute (contextualized) value vector for each "head"
        values, attn = self.scaled_dot_product(q, k, v, src_padding_mask, future_mask)

        # Concatenate contextualized value vectors from all heads
        values = values.reshape(batch_size, sequence_length, hidden_dim)

        # Linearly transform the concatenation of all heads' value vectors (8*64=512) to the original hidden dim (512)
        output = self.o_proj(values)
        return output

    def _self_attention_projection(self, x: torch.FloatTensor):
        """
        Project x and interpret the result as chunks that represent q, k and v vectors for every head.
        Input x can be encoder or decoder hidden states, depending on which one calls this MHA module.

        :param x: Encoder or decoder hidden states. (batch_size, seq_len, hidden_dim)
        :return: query, key and value vectors. (batch_size, seq_len, num_heads, hidden_dim // num_heads)
        """
        batch_size, sequence_length, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        return q, k, v

    def _cross_attention_projection(
        self,
        encoder_hidden_states: torch.FloatTensor,
        decoder_hidden_states: torch.FloatTensor,
    ):
        """
        Projects decoder hidden states into query vectors and encoder hidden states into key and value vectors.
        The columns of W_proj determine how much independent linear combinations of the input we obtain - which we
        then interpret as heads and qkv vectors. Thus we can simply split the weight matrix and project the decoder
        hidden states x into q separately from projecting the encoder_hidden_states into k and v.

        :param encoder_hidden_states: (batch_size, src_seq_len, hidden_dim)
        :param decoder_hidden_states: (batch_size, tgt_seq_len, hidden_dim)
        :return: query vector (batch_size, tgt_seq_len, num_heads, qkv_dim) and
            key and value vectors (batch_size, src_seq_len, num_heads, qkv_dim)
        """
        batch_size, src_sequence_length, hidden_dim = encoder_hidden_states.shape
        batch_size, tgt_sequence_length, hidden_dim = decoder_hidden_states.shape

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
            .reshape(batch_size, src_sequence_length, self.num_heads, 2 * self.qkv_dim)
            .chunk(2, dim=-1)
        )

        # Project decoder hidden states into q's
        q = F.linear(input=decoder_hidden_states, weight=w_q, bias=b_q).reshape(
            batch_size, tgt_sequence_length, self.num_heads, self.qkv_dim
        )

        return q, k, v

    def scaled_dot_product(
        self,
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        For cross-attention, the sequence length of q and (k,v) may differ as q is projected from decoder hidden states
        and kv from encoder hidden states.

        :param q: (batch_size, num_heads, seq_len, qkv_dim)
        :param k: (batch_size, num_heads, seq_len, qkv_dim)
        :param v: (batch_size, num_heads, seq_len, qkv_dim)
        :param src_padding_mask: (batch_size, src_seq_len)
        :param future_mask: (tgt_seq_len, tgt_seq_len)
        :return: values (batch_size, num_heads, seq_len, qkv_dim), attention (batch_size, num_heads, seq_len, seq_len)
        """

        # Compute attention logits. Dot product between each query and key vector, through one matrix multiplication.
        # Results in un-normalized attention scores for each position's query vector to each position's key vector
        # Result is (batch_size, num_heads, seq_length, seq_length)
        attn_logits = torch.matmul(q, torch.transpose(k, -2, -1),)

        # Scale logits by constant to create less spiky softmax distribution
        attn_logits = attn_logits / math.sqrt(q.size()[-1])

        # Apply attention mask (for pad tokens and future-masking in cross-attention)
        if src_padding_mask is not None or future_mask is not None:
            attn_logits = self.mask_logits(attn_logits, src_padding_mask, future_mask)

        # Transform logits to attention probability distribution (one distribution per non-masked token index)
        attention = F.softmax(attn_logits, dim=-1)

        # Weighted sum of value vectors for each input token using attention scores -> new contextualized representation
        # (batch_size, num_heads, sequence_length, qkv_dim)
        values = torch.matmul(attention, v)
        return values, attention

    @staticmethod
    def mask_logits(
        logits: torch.FloatTensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Reshape masks to fit the shape of the logits and set all indices with "False" to -inf

        :param logits: (batch_size, num_heads, seq_length, seq_length)
        :param src_padding_mask: (batch_size, src_seq_len)
        :param future_mask: (tgt_seq_len, tgt_seq_len)
        :return: masked_logits (batch_size, num_heads, seq_length, seq_length)
        """
        if src_padding_mask is not None:
            masked_logits = logits.masked_fill(
                src_padding_mask[:, None, None, :] == 0, float("-inf")
            )
        if future_mask is not None:
            masked_logits = logits.masked_fill(future_mask == 0, float("-inf"))
        return masked_logits


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

    def test_scaled_dot_product_encoder_self_attention_mask(self):
        mha = MultiHeadAttention(hidden_dim=512, num_heads=8)
        q = torch.randn(2, 8, 10, 512, dtype=torch.float)
        k = torch.randn(2, 8, 10, 512, dtype=torch.float)
        v = torch.randn(2, 8, 10, 512, dtype=torch.float)
        mask = torch.BoolTensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )

        _, attention_scores = mha.scaled_dot_product(q, k, v, src_padding_mask=mask)
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

    def test_future_masking(self):
        batch_size, n_heads, seq_len = 2, 2, 3  # TODO add 2 heads and batch_size=3
        logits = torch.randn(batch_size, n_heads, seq_len, seq_len, dtype=torch.float)
        future_mask = construct_future_mask(seq_len)
        self.assertEqual(future_mask.shape, (3, 3))

        masked_logits = MultiHeadAttention(512, num_heads=n_heads).mask_logits(
            logits, future_mask=future_mask
        )
        torch.testing.assert_close(
            torch.isinf(masked_logits) == 0,
            torch.BoolTensor(
                [
                    [
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                    ],
                    [
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                        [
                            [True, False, False],
                            [True, True, False],
                            [True, True, True],
                        ],
                    ],
                ]
            ),
        )

    def test_src_padding_masking(self):
        batch_size, n_heads, seq_len = 2, 2, 3
        logits = torch.randn(batch_size, n_heads, seq_len, seq_len, dtype=torch.float)
        src_padding_mask = torch.BoolTensor([[True, True, True], [True, False, False]])
        self.assertEqual(src_padding_mask.shape, (2, 3))
        masked_logits = MultiHeadAttention(512, num_heads=n_heads).mask_logits(
            logits, src_padding_mask=src_padding_mask
        )
        torch.testing.assert_close(
            torch.isinf(masked_logits) == 0,
            torch.BoolTensor(
                [
                    [
                        [[True, True, True], [True, True, True], [True, True, True],],
                        [[True, True, True], [True, True, True], [True, True, True],],
                    ],
                    [
                        [
                            [True, False, False],
                            [True, False, False],
                            [True, False, False],
                        ],
                        [
                            [True, False, False],
                            [True, False, False],
                            [True, False, False],
                        ],
                    ],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
