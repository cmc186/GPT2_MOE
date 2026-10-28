# Basic Imports
import torch

# Relative imports
from .common import IGNORE_INDEX, per_task


def token_accuracy(model, loader, device:str, **kwargs)->dict:
    """
        Function aimed to compute the accuracy over the
        answer tokens (teacher forcing) per task

        Params:
            model(GPTMOE):
                The model to evaluate
            loader(DataLoader):
                Loader over a TaskDataset
            device(str):
                Device: cuda or cpu

        Returns:
            results(dict):
                {task_name: accuracy, ..., 'mean': macro mean}
    """
    names = loader.dataset.task_names
    correct = torch.zeros(len(names))
    total = torch.zeros(len(names))

    for batch in loader:
        inputs = batch['input_ids'].to(device)
        targets = batch['target_ids'].to(device)
        tasks = batch['task']
        # Only the answer and <|endoftext|> count (B, N)
        mask = targets != IGNORE_INDEX
        hits = (model(inputs).argmax(-1) == targets) & mask
        # Accumulate per example into its task
        correct.index_add_(0, tasks, hits.sum(-1).float().cpu())
        total.index_add_(0, tasks, mask.sum(-1).float().cpu())

    return per_task(correct, total, names)
