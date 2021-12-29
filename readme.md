# Custom implementation of Vanilla Transformers
As an exercise I implemented the vanilla encoder-decoder transformer 
described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762) without any dependencies except PyTorch. 

WORK IN PROGRESS!

## Features:
- The simplest imaginable Vocabulary class (vocabulary.py)
- The simplest imaginable tokenization method (vocabulary.py)
- TransformerEncoder class (transformer.py)
- TransformerDecoder class (transformer.py)
- Transformer class (transformer.py)
- MultiHeadAttention class (multi_head_attention.py)
- SinusoidEncoding class (positional_encoding.py)
- Basic unit tests for each class
- Code formatted using [black](https://github.com/psf/black)
- Python 3.9, PyTorch 1.9.1

## To do:
- Finish TransformerDecoder implementation: cross & self-attention
- Decoder & Transformer main class unit tests
- Attention mask for pad tokens and decoder self-attention
- Divide embedding weights by sqrt(hidden_dim // num_heads)
- Training loop, optimizer & settings etc.
  - Check whether BOS and EOS tokens should be added to each training sentence; remove this from the vocab class if not.

#### Bonus
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support
- Run unit tests on each commit using Github Actions 
  - Refactor unit tests to separate folder and files
  - Run pylint on codebase
- Publish this repo and write blogpost about what I learned / what surprised me.

## Lessons Learned / Implementation Notes
- The query, key and value vector dim is dynamically set to hidden_dim/num_heads in most (e.g. PyTorch & huggingface) default implementations <include source>.
- Multi-head attention is parameterized by one weight matrix per layer. For example, say the input is of dim(1, 10, 512), then W_proj_qkv has dim(512, 3*512). This results in
3 times 512 linear combinations of each row in x. We can further interpret each block of 512 as 8 blocks of 64 that represent the “heads”.
- Vanilla transformers can handle arbitrary length inputs in theory. In practice, however, self-attention has [compute and memory requirements](https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html#:~:text=With%20commonly%20available%20current%20hardware,summarization%20or%20genome%20fragment%20classification.) that limit the input sequence to ~512 tokens. Models like BERT do impose a hard length limit because they use learned embeddings instead of the sinusoid encoding, which stop existing after a certain length (512).
- Each input sequence in a batch must be padded to the longest sentence in the batch: it's common practice to batch sequences together with similar length.
- Positional encodings add different sine (even dims) or cosine (uneven dims) waves to each dimension that "flow across" tokens
- The embedding weights are shared between encoder and decoder, and its transpose is used as the pre-softmax linear transformation to reduce overfitting (?) and # params. <todo: why>?
- The final/last/top hidden state that corresponds to the decoder’s last input token is used to predict the next token: it's used as input to the above linear transformation that produces logits (an unnormalized distribution) over the vocabulary. The other hidden states seem to be simply discarded?
- The embedding weights are divided by sqrt(hidden_dim // num_heads). <todo: why?> 

## References
- For positional encodings and the MHA implementations looked at (and replaced my own implementation with) snippets from this great tutorial: 
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- Thanks David Stap for suggesting me to implement the Attention is all you need paper from scratch!