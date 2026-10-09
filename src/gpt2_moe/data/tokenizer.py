# Import modules
import torch
import torch.nn as nn
import tiktoken

class Tokenizer:
    '''
    Wrapper class for the GPT-2 tokenizer using the tiktoken library.
    '''
    def __init__(self, embedding_dim: int, max_seq_len: int):
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab

    def encode(self, text: str) -> list[int]:
        '''
        Encode a string into a list of token IDs.
        Parameters:
        -----------
            text (str): The input string to encode
        Returns:
        --------
            list[int]: A list of token IDs corresponding to the input string
        '''
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        '''
        Decode a list of token IDs back into a string.
        Parameters:
        -----------
            ids (list[int]): A list of token IDs to decode
        Returns:
        --------
            str: The decoded string
        '''
        return self.tokenizer.decode(ids)

