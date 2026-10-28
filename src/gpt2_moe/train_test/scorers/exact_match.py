# Basic Imports
import torch
import time

# Relative imports
from .common import per_task, split_example, greedy_answer


def exact_match(model:object, loader:object, device:str, max_per_task:int = None, **kwargs)->dict[str:float]:
    """
        Function aimed to compute the exact match of the
        generated answer per task

        Params:
            model(GPTMOE):
                The model to evaluate
            loader(DataLoader):
                Loader over a TaskDataset
            device (str):
                The selected device
            max_per_task(int):
                Max examples evaluated per task.
                Default value is None -> all

        Returns:
            results(dict):
                A dict composed by task:score pairs and also includes mean inference time
    """
    # Get the dataset
    ds = loader.dataset
    # Get the task names
    names = ds.task_names
    # Define scoring trackers
    correct = torch.zeros(len(names))
    total = torch.zeros(len(names))

    # Define time estimation
    timer = []

    # Iterate over batches
    for batch in loader:
        # Iterate within batches
        for input, target, task in zip(batch['input_ids'], batch['target_ids'], batch['task'].tolist()):
            # Define per task token eval thresh
            if max_per_task is not None and total[task] >= max_per_task:
                continue
            # Retrieve the  clean prompt and the answer
            prompt, answer = split_example(input, target)
            # Generate and decode with greedy method
            start = time.time()
            prediction = greedy_answer(model, prompt.to(device), ds.eos_id, max_new = len(answer) + 5)
            timer.append(time.time() - start)
            # Compare decoded strings without spaces -> exact matching
            matching = ds.tokenizer.decode(prediction).strip() == ds.tokenizer.decode(answer.tolist()).strip()
            # Register per class counter
            correct[task] += int(matching)
            # Increase task counter for later aggregated metrics
            total[task] += 1

    # Get result dict
    result = per_task(correct, total, names)
    # Add mean inference time
    result['mean_time'] = sum(timer) / len(timer) 

    return result
