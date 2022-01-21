import math
import torch
from torch import nn
import unittest


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
    def scaled_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """ All input tensors have dimensionality (n_batches, num_heads, sequence_length, qkv_dim) """

        # Dot product between all query vectors and all key vectors
        # E.g.: first entry contains the dot product between the first query vector and all key vectors in the sequence
        attn_logits = torch.matmul(
            q, torch.transpose(k, -2, -1)
        )  # (n_batches, num_heads, seq_length, seq_length)

        # Scale these "cross dot products" by a constant
        attn_logits_scaled = attn_logits / math.sqrt(q.size()[-1])

        # Transform logits to attention probability distribution
        attention = nn.functional.softmax(attn_logits_scaled, dim=-1)

        # Weighted sum of value vectors for each input token using attention weights -> contextualized representation
        values = torch.matmul(attention, v)  # (n_batches, num_heads, sequence_length, qkv_dim)
        return values, attention

    def forward(self, x: torch.Tensor):
        """ Perform multi-head attention"""
        n_batches, sequence_length, hidden_dim = x.size()

        # Extract Q, K and V
        qkv = self.qkv_proj(x)  # why can't I simply do .chunk(3, dim=-1) here?
        qkv = qkv.reshape(n_batches, sequence_length, self.num_heads, 3 * self.qkv_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (n_batches, num_heads, sequence_length, 3*qkv_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # Compute contextualized value vector for each "head"
        values, attn = self.scaled_dot_product(q, k, v)  # (n_batches, num_heads, seq_len, qkv_dim)

        # Concatenate contextualized value vectors from all heads
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(n_batches, sequence_length, hidden_dim)

        output = self.o_proj(values)
        return output


class TestMultiHeadAttention(unittest.TestCase):
    def test_scaled_dot_product(self):
        mha = MultiHeadAttention(512, 8)
        q = torch.randn(4, 8, 10, 512)
        k = torch.randn(4, 8, 10, 512)
        v = torch.randn(4, 8, 10, 512)

        values, attention = mha.scaled_dot_product(q, k, v)

        self.assertEqual(values.shape, (4, 8, 10, 512))
        self.assertEqual(attention.shape, (4, 8, 10, 10))
        self.assertEqual(True in torch.isnan(values), False)
        self.assertEqual(True in torch.isnan(attention), False)

    def test_forward(self):
        mha = MultiHeadAttention(512, 8)
        x = torch.randn(4, 10, 512)
        output = mha.forward(x)
        self.assertEqual(output.shape, (4, 10, 512))
        self.assertEqual(True in torch.isnan(output), False)


if __name__ == "__main__":
    unittest.main()
