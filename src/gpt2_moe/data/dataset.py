# Import modules
from torch.utils.data import Dataset
import torch
import pandas as pd
import os

from gpt2_moe.data.tokenizer import Tokenizer


# Define dir for data
DATA_DIR = "data" 



class TaskDataset(Dataset):

    # Ignored for CrossEntropyLoss (optionally)
    IGNORE_INDEX = -100

    def __init__(self, tokenizer: Tokenizer, context_lenght: int, seed: int = 42, fractions: tuple = (0.8, 0.1, 0.1), datasets: list = [], split: str = "train"):
        super().__init__()

        self.tokenizer = tokenizer
        self.context_lenght = context_lenght
        self.seed = seed
        self.splits = dict(zip(["train", "val", "test"], fractions))
        self.split = split
        self.datasets = datasets 
        self.current = 0

        # <|endoftext|>
        self.eos_id = tokenizer.tokenizer.eot_token

        self.rows_per_csv: dict[str, int] = {}
        self.examples: list[tuple[list[int], list[int], int]] = []

        # Read all the CSVs into one table (prompt, y, task)
        df = self._load_csvs()
        self.task_names: list[str] = sorted(df["task"].unique())

        # Keep only the rows of this split
        df = self._split(df)

        # Tokenize every row once
        for prompt, y, task in zip(df["prompt"], df["y"], df["task"]):
            example = self._encode_example(prompt, y, task)
            if example is not None:
                self.examples.append(example)

    def _load_csvs(self) -> pd.DataFrame:
        '''
        Load all CSV files into a single DataFrame
        returns:
        --------
            pd.DataFrame: A DataFrame containing all the data from the CSV files
        '''
        # Check if there are any CSV files
        if not self.datasets:
            raise FileNotFoundError(f"No CSV found in {DATA_DIR}")

        # Read all CSVs
        frames = []
        for self.current, path in enumerate(self.datasets):

            # Read everything as text
            df = pd.read_csv(path, dtype=str)
            # Add a column with the task name
            df["task"] = os.path.splitext(os.path.basename(path))[0]

            # Store rows per csv
            self.rows_per_csv[path] = len(df)
            frames.append(df)

        # Remove repeated rows so train and val never share examples
        return pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)

    def _split(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Split the DataFrame into train, val, and test
        Parameters:
        -----------
            df (pd.DataFrame): The DataFrame to split
        Returns:
        --------
            pd.DataFrame: A DataFrame containing only the rows of the specified split
        '''
        # Get split names
        names = list(self.splits)
        # Get index por work-split
        i = names.index(self.split)
        # Get cumulative proportion until :selected_split
        start_frac = sum(self.splits[name] for name in names[:i])
        end_frac = start_frac + self.splits[self.split]
        # Check if it is the last split
        is_last = (i == len(names) - 1)

        parts = []
        # Each task is split separately so every task keeps the fractions
        for _, group in df.groupby("task"):
            # Shuffles selected rows 
            group = group.sample(frac = 1.0, random_state = self.seed)
            # Get indices start:end   
            n = len(group)
            start = int(n * start_frac)
            # The last split takes everything left
            end = n if is_last else int(n * end_frac)
            parts.append(group.iloc[start:end])
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def build_prompt(prompt: str, task: str) -> str:
        '''
        Build a prompt string for the model
        
        Parameters:
        -----------
            prompt (str): The prompt to build
            task (str): The task to build the prompt for

        Returns:
        --------
            str: The built prompt
        '''
        return f"{task}: {prompt} ="

    @staticmethod
    def build_answer(y: str) -> str:
        '''
        Build an answer string for the model    
        Parameters:
        -----------
            y (str): The answer to build

        Returns:
        --------
            str: The built answer
        '''
        return f" {y}"

    def _encode_example(self, prompt: str, y: str, task: str):
        '''
        Encode a single example into input and target IDs for the model
        Parameters:
        -----------
            prompt (str): The prompt to encode
            y (str): The answer to encode
            task (str): The task to encode

        Returns:
        --------
            tuple[list[int], list[int], int]: A tuple containing the input IDs, target IDs, and task index
        '''
        # Encode prompt and answer separately to know where the answer starts
        prompt_ids = self.tokenizer.encode(self.build_prompt(prompt, task))
        answer_ids = self.tokenizer.encode(self.build_answer(y)) + [self.eos_id]
        seq = prompt_ids + answer_ids

        # Too long for the model (input is len(seq) - 1)
        if len(seq) - 1 > self.context_lenght:
            return None

        # Shift by one token: position t predicts token t+1
        input_ids = seq[:-1]
        target_ids = seq[1:]

        # Mask the prompt: only the answer and <|endoftext|> count in the loss
        n_masked = len(prompt_ids) - 1
        target_ids = [self.IGNORE_INDEX] * n_masked + target_ids[n_masked:]

        return input_ids, target_ids, self.task_names.index(task)

    def __len__(self):
        '''Return the number of examples in the dataset'''
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        '''
        Return a single example from the dataset
        Parameters:
        -----------
            idx (int): The index of the example to return
        Returns:
        --------
            dict[str, torch.Tensor | int]: A dictionary containing the input IDs, target IDs, and task index
        '''
        # Retrieve the previously created prompts
        input_ids, target_ids, task = self.examples[idx]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "task": task,
        }