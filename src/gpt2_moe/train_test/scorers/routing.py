# Basic Imports
import torch

# Relative imports
from .common import IGNORE_INDEX
from .jenser_divergence import JensenDivergence


def route(model, input_ids:torch.Tensor, device:str)->dict[int, torch.Tensor]:
    """
        Function aimed to retrieve the experts selected
        by the router of every MoE layer for each token

        Params:
            model(GPTMOE):
                The model to inspect
            input_ids(torch.Tensor):
                Input ids (B, N)
            device(str):
                The selected device

        Returns:
            routes(dict):
                A dict representing router probability
                output (B, N, K) per moed layer
    """
    # Get the MoE layer indices -> (0, 2, 4, ...)
    layers = [i for i, use_moe in enumerate(model.moe_mask) if use_moe]
    # Get input dims (B, N)
    n_batches, tokens = input_ids.shape

    # Restore the routing and feed the model
    model.eval()
    model.set_forced_expert(None)
    with torch.no_grad():
        model(input_ids.to(device = device))

    # Retrieve the routing state (B*N, k) -> (B, N, k)
    return {layer: moe.last_selected.view(n_batches, tokens, -1).to(device = 'cpu')
            for layer, moe in zip(layers, model.moe_layers())}


def real_tokens(target_ids:torch.Tensor)->torch.Tensor:
    """
        Function aimed to build the mask of the non padded
        positions (prompt, answer and <|endoftext|>)

        Params:
            target_ids(torch.Tensor):
                Padded target ids (B, N)

        Returns:
            mask(torch.Tensor):
                True for real tokens (B, N)
    """
    # Position of every token (N,)
    positions = torch.arange(target_ids.shape[-1])
    # Last answered position per example -> <|endoftext|> target (B,)
    last = torch.where(target_ids != IGNORE_INDEX, positions, -1).max(dim = -1).values
    # Real tokens are the ones up to it (B, N) -> tak broadcasting
    return positions.unsqueeze(0) <= last.unsqueeze(-1)


def routing_matrix(model:object, loader:object, device:str)->torch.Tensor:
    """
        Function aimed to compute how often the tokens of
        each task are routed to each expert per MoE layer

        Params:
            model(GPTMOE):
                The model to inspect
            loader(DataLoader):
                Loader over a TaskDataset
            device(str):
                The selected device

        Returns:
            matrix(torch.Tensor):
                Routing distribution (layers, tasks, experts)
                Each (layer, task) row sums one
    """
    # Get task names and the MoE modules
    names = loader.dataset.task_names
    moes = model.moe_layers()
    n_exp = moes[0].n_exp

    # Define the counter (layers, tasks, experts)
    counts = torch.zeros(size = (len(moes), len(names), n_exp))

    # Iterate over batches
    for batch in loader:
        # Get the routing per layer {layer: (B, N, k)}
        routes = route(model, batch['input_ids'], device)
        # Keep only real tokens (B, N) -> set -100 to false
        mask = real_tokens(batch['target_ids'])

        # Iterate per layer
        for i, selected in enumerate(routes.values()):
            # Get shapes
            B, N, K = selected.shape
            # Pair each assignation with its task (B, N, k)
            task = batch['task'].view(B, 1, 1).repeat(1, N, K)
            # Mask shape (B, N, 1) -> (B, N, K)
            valid = mask.unsqueeze(-1).repeat(1, 1, K)
            # Gen valid task expert pairs
            # 1. Define expert_sized range per task -> [0, experts] unique identifier within task
            expert_sized_range = task * n_exp
            # 2. Generate global index
            raw_pair = (expert_sized_range + selected)
            # 3. Filter not valid pairs (-100 marked) -> (sum(mask))
            pairs = raw_pair[valid]
            # Populate layer: add vector counting [bin0, bin1, ..., bink] each bin belongs to an expert in a given task
            raw_count = torch.bincount(pairs, minlength = len(names) * n_exp)
            # Reshape into (tasks, experts)
            polished_count = raw_count.view(len(names), n_exp).float()
            # Increase layer count
            counts[i]+=polished_count

    # Normalize  to get a probability distro
    return torch.divide(counts, counts.sum(dim = -1, keepdim = True))


def routing_jsd(model, loader, device:str, **kwargs)->dict[str:float]:
    """
        Function aimed to measure expert specialization as the
        mean Jensen-Shannon divergence between the routing
        distributions of every pair of tasks

        Values are normalized to [0, 1]:
            0 -> every task uses the same experts
            1 -> tasks use disjoint experts

        Params:
            model(GPTMOE):
                The model to evaluate
            loader(DataLoader):
                Loader over a TaskDataset
            device(str):
                The selected device

        Returns:
            results(dict):
                {layer_i: jsd, ..., 'mean': mean over layers}
                Empty dict for dense models (no router)
    """
    # Dense models have no router
    if not model.moe_layers():
        return {}

    # Get the routing distributions (layers, tasks, experts)
    matrix = routing_matrix(model, loader, device)
    layers = [i for i, use_moe in enumerate(model.moe_mask) if use_moe]
    # Pairs of different tasks -> upper triangle
    n_tasks = matrix.shape[1]
    rows, cols = torch.triu_indices(n_tasks, n_tasks, offset = 1)

    results = {}
    for layer, dist in zip(layers, matrix):
        # JSD between every pair of tasks (T, T) normalized by ln(2)
        jsd = JensenDivergence()(dist.unsqueeze(1), dist.unsqueeze(0)) / torch.log(torch.tensor(2.0))
        results[f'layer_{layer}'] = jsd[rows, cols].mean().item()

    # Compute the general mean
    results['mean'] = sum(results.values()) / len(results)
    return results
