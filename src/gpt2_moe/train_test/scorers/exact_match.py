# Basic Imports
import torch

# Relative imports
from .common import per_task, split_example, greedy_answer


@torch.no_grad()
def exact_match(model, loader, device:str, max_per_task:int | None = None, **kwargs)->dict:
    """
        Function aimed to compute the exact match of the
        generated answer per task

        Params:
            model(GPTMOE):
                The model to evaluate
            loader(DataLoader):
                Loader over a TaskDataset
            device(str):
                cuda or cpu
            max_per_task(int | None):
                Max examples evaluated per task. None -> all

        Returns:
            results(dict):
                {task_name: exact match, ..., 'mean': macro mean}
    """
    ds = loader.dataset
    names = ds.task_names
    correct = torch.zeros(len(names))
    total = torch.zeros(len(names))

    for batch in loader:
        for inp, tgt, task in zip(batch['input_ids'], batch['target_ids'], batch['task'].tolist()):
            # Skip tasks that already reached the limit
            if max_per_task is not None and total[task] >= max_per_task:
                continue
            prompt, answer = split_example(inp, tgt)
            pred = greedy_answer(model, prompt.to(device), ds.eos_id, max_new = len(answer) + 5)
            # Compare strings: the same text can be tokenized differently
            ok = ds.tokenizer.decode(pred).strip() == ds.tokenizer.decode(answer.tolist()).strip()
            correct[task] += int(ok)
            total[task] += 1

    return per_task(correct, total, names)
