# Import modules
import os
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from gpt2_moe.data.dataset import TaskDataset
from gpt2_moe.data.tokenizer import Tokenizer

DATA_DIR = "data" 

class DataLoader:
    def __init__(self, tokenizer: Tokenizer, context_lenght: int, batch_size: int, seed: int = 42, fractions: tuple = (0.8, 0.1, 0.1), datasets: list = [], max_lenght: int | None = None):
        """
            Build the train, val and test dataloaders with every example padded to the same length
            Params:
                tokenizer (Tokenizer): 
                    Tokenizer used to encode the examples
                context_lenght (int): 
                    Max length the model accepts
                batch_size (int): 
                    Examples per batch
                seed (int): 
                    Seed for the split and the shuffle
                fractions (tuple): 
                    Fractions for train, val and test
                datasets (list): 
                    CSV paths. Empty -> every CSV inside data/
                max_lenght (int | None): 
                    Padded length. None -> longest example of the training set
        """
        self.tokenizer = tokenizer
        self.context_lenght = context_lenght
        self.batch_size = batch_size
        self.seed = seed
        self.splits = dict(zip(["train", "val", "test"], fractions))
        self.datasets = datasets 
        self.max_lenght = max_lenght

        # Same CSVs and seed for the three splits, they never share rows
        self.train_ds, self.val_ds, self.test_ds = [
            TaskDataset(tokenizer, context_lenght, seed=seed, fractions=tuple(self.splits.values()),
                        datasets=self.datasets, split=split)
            for split in self.splits
        ]

        # <|endoftext|> is also used as padding
        self.pad_id = self.train_ds.eos_id

        # If none, use the longest example of the training set
        if self.max_lenght is None:
            self.max_lenght = self._longest(self.train_ds)
        self._check_lengths()

        # Pad every example once
        for ds in (self.train_ds, self.val_ds, self.test_ds):
            self._pad(ds)

        # Reproducible shuffle of the training set
        generator = torch.Generator().manual_seed(seed)

        self.train_loader = TorchDataLoader(self.train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=generator)
        self.val_loader = TorchDataLoader(self.val_ds, batch_size=batch_size, shuffle=False)
        self.test_loader = TorchDataLoader(self.test_ds, batch_size=batch_size, shuffle=False)

    @staticmethod
    def _longest(ds: TaskDataset) -> int:
        """
            Length of the longest example in a dataset
            Params:
                ds (TaskDataset): 
                    The dataset to measure
            Returns:
                int: 
                    Number of tokens of its longest input_ids
        """
        return max(len(input_ids) for input_ids, _, _ in ds.examples)

    def _check_lengths(self) -> None:
        """
            Make sure max_lenght fits in the model and every example fits in max_lenght
        """
        if self.max_lenght > self.context_lenght:
            raise ValueError(f"max_lenght={self.max_lenght} is bigger than context_lenght={self.context_lenght}")

        for _, ds in zip(self.splits, (self.train_ds, self.val_ds, self.test_ds)):
            longest = self._longest(ds)
            if longest > self.max_lenght:
                raise ValueError(f"Example longer than max_lenght")

    def _pad(self, ds: TaskDataset) -> None:
        """

            Pad on the right every example of a dataset up to max_lenght
            Params:
                ds (TaskDataset): 
                    The dataset to pad 
        """
        padded = []
        for input_ids, target_ids, task in ds.examples:
            n_pad = self.max_lenght - len(input_ids)
            padded.append((
                input_ids + [self.pad_id] * n_pad,
                target_ids + [TaskDataset.IGNORE_INDEX] * n_pad,
                task,
            ))
        ds.examples = padded

    def get_loaders(self) -> tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
            Function aimed to get loaders
            Returns:
                tuple[DataLoader, DataLoader, DataLoader]:
                    Train, val and test dataloaders
        """

        return self.train_loader, self.val_loader, self.test_loader


