
# Implementing A Transformer From Scratch
To get intimately familiar with the nuts and bolts of transformers, I implemented a bare-bone version of the original transformer 
proposed in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). No dependencies except [PyTorch](https://pytorch.org/get-started/locally/) and not intended for real-world use.

WORK IN PROGRESS!

# To do:
### Code
- The encoder hidden states seem to be too small which somehow causes the decoder to place most of its mass on 0th index for each hidden state for each batch instance. I can fix this by scaling the encoder hidden states with sqrt(hidden_dim) but this should not be happening.
- Write minimal training loop to test if model can fit a tiny dataset
  - Optimizer & settings
  - Check whether BOS and EOS tokens should be added to each training sentence; remove this from the tokenization if not.

### Writing
- Add details about masking in MHA
- Add details about the formal meaning of the word projection
- Expand on layer normalization; most people know it normalizes the batch somehow but what does that actually do?
- Write about relation between transformer and popular models such as bert or gpt
- Think about title, motivation and framing

# Features:
- The simplest imaginable Vocabulary class (vocabulary.py)
- The simplest imaginable (batch) tokenization (vocabulary.py)
- TransformerEncoder and EncoderBlock class (encoder.py)
- TransformerDecoder and EncoderBlock class (decoder.py)
- Transformer class (transformer.py)
- MultiHeadAttention class with scaled dot product method (multi_head_attention.py)
- SinusoidEncoding class (positional_encoding.py)
- Basic unit tests for each class
- Type checking
- Code formatted using [black](https://github.com/psf/black)
- Python 3.9, PyTorch 1.9.1

### Bonus to do
- Replace examples in the vocabulary unit test.
- (Multi-)GPU support
- Run unit tests on each commit using Github Actions 
  - Refactor unit tests to separate folder and files
  - Run pylint on codebase