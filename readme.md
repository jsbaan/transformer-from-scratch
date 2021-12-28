# Custom implementation of Vanilla Transformers
As an exercise I implemented the vanilla encoder-decoder transformer 
described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

WORK IN PROGRESS!

Features:
- The simplest possible tokenizer
- The simplest possible vocabulary
- TransformerEncoder class
- TransformerDecoder class
- Transformer class
- MultiHeadAttention class
- SinusoidEncoding class
- Simple unit tests for each class
- Code formatted using [black](https://github.com/psf/black)

## Implementation Notes
- The query, key and value vector dim is dynamically set to hidden_dim/num_heads.
- Multi-head attention is parameterized by one linear layer. For example, say the input is of dim(1, 10, 512), then W_proj_qkv has dim(512, 3*512). This results in
3 times 512 linear combinations of each row in x. We can further interpret each block of 512 as 8 blocks of 64 that represent the “heads”.
- Vanilla transformers can handle arbitrary length inputs in theory. [In practice](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html#:~:text=With%20commonly%20available%20current%20hardware,summarization%20or%20genome%20fragment%20classification.), however, self-attention has compute and memory requirements that limit the input sequence to ~512 tokens. Models like BERT do impose a hard length limit because they use learned embeddings instead of the sinusoid encoding, which stop existing after a certain length (512).
- Each input sequence in a batch must be padded to the longest sentence in the batch.
- Positional encodings add different sine or cosine waves per dimension over input length
- The embedding weights are shared between encoder and decoder, and its transpose is used as the pre-softmax linear transformation to reduce overfitting (?) and # params.
- The final hidden state corresponding to the decoder’s last input token is used as input to the above linear transformation that produces logits over the vocabulary. The other hidden states seem to be simply discarded?

## References
- For positional encodings and the MHA implementations I cheated
and looked at this great tutorial: 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
I also looked
- Thanks to David Stap for suggesting me to do this!