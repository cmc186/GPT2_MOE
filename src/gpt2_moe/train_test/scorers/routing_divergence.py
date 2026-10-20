# Basic Imports
import math
import torch

# Relative imports
from .common import IGNORE_INDEX
from .jenser_divergence import JensenDivergence


@torch.no_grad()
def routing_divergence(model, loader, device:str, **kwargs)->dict:
    """
        Function aimed to measure router specialization.
        For every MoE layer a task x expert distribution is
        built over the answer tokens and the mean pairwise
        JSD between tasks is returned, normalized to [0, 1]

        Params:
            model(GPTMOE):
                The model to evaluate
            loader(DataLoader):
                Loader over a TaskDataset
            device(str):
                cuda or cpu

        Returns:
            results(dict):
                {'layer_i': jsd, ..., 'mean': mean over layers}
                Empty if there is no MoE or an expert is forced
    """
    moes = model.moe_layers()
    # No routing to measure
    if not moes or moes[0].forced_expert is not None:
        return {}

    names = loader.dataset.task_names
    n_exp = moes[0].n_exp
    # (layers, tasks, experts)
    counts = torch.zeros(len(moes), len(names), n_exp)

    for batch in loader:
        inputs = batch['input_ids'].to(device)
        targets = batch['target_ids'].flatten()
        # Forward only to fill last_selected
        model(inputs)
        # Task of every token (B*N,) and answer mask
        tasks = batch['task'].repeat_interleave(inputs.shape[1])
        mask = targets != IGNORE_INDEX
        tasks = tasks[mask]
        ones = torch.ones(len(tasks))
        for l, moe in enumerate(moes):
            # (T, k)
            selected = moe.last_selected.cpu()[mask]
            for j in range(selected.shape[1]):
                counts[l].index_put_((tasks, selected[:, j]), ones, accumulate = True)

    jsd = JensenDivergence()
    results = {}
    for l in range(len(moes)):
        # Keep tasks present in the loader
        present = counts[l].sum(-1) > 0
        if present.sum() < 2:
            continue
        dist = counts[l][present] / counts[l][present].sum(-1, keepdim = True)
        # Pairwise JSD (T, T) normalized by ln2
        pairwise = jsd(dist.unsqueeze(1), dist.unsqueeze(0)) / math.log(2)
        i, j = torch.triu_indices(len(dist), len(dist), offset = 1)
        results[f'layer_{l}'] = pairwise[i, j].mean().item()

    if results:
        results['mean'] = sum(results.values()) / len(results)
    return results
