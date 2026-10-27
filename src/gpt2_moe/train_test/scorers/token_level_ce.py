# Basic import
import torch
import torch.nn as nn


def token_level_ce(model:object, loader, device:str, **kwargs)->dict[str:float]:
    """
        Function aimed to compute per token CE

        Params:
            model(GPTMOE):
                The model to evaluate
            loader(DataLoader):
                Loader over a TaskDataset
            device (str):
                The selected device

        Returns:
            results(dict):
                A dict composed by task:score items
    """
    # Get the dataset and task names
    ds = loader.dataset
    names = ds.task_names
    n = len(loader)

    # Define the loss function
    loss = nn.CrossEntropyLoss(
        ignore_index = -100,
        reduction = 'none'
    )

    # Define storages
    loss_per_task = torch.zeros(len(names), dtype=torch.float64)
    tokens_per_task = torch.zeros(len(names), dtype=torch.long)

    # Set model to eval
    model.eval()
    with torch.no_grad():
        # Iterate over batches
        for batch in loader:
            input = batch['input_ids'].to(device = device)
            target = batch['target_ids'].to(device = device)
            task = batch['task'].to(device = device)
            # Generate prediction (B, S, |V|)
            logits = model(input)
            batch_size, seq_len, vocab_size = logits.shape
            #print('(ce_t) Prediction aspect:', logits.shape)
            #print('(ce_t) Answer shape:', target.shape)
            #print('(ce_t) Task shape:', task.shape)

            # Apply CE (B*S, |V|) -> get loss per token -> (B, S)
            tokens_loss = loss(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)).reshape(batch_size, seq_len)
            #print('(ce_t) Loss Shape', tokens_loss.shape)
            
            # Filter tokens
            valid_mask = target!=-100
            print('(ce_t) Aspect of the mask:', valid_mask.shape)
         
            # Filter tasks (task is a mask)
            for task_id, name in enumerate(names):
                # Get examples from batch which belong to task
                task_mask = task==task_id
                #print(F'     (ce_t)({name}) Aspect of the task mask:', valid_mask.shape)

                # Apply (B, S) and (B, 1) -->(B,S )
                selected_mask = torch.bitwise_and(valid_mask, task_mask.unsqueeze(1))

                # Populate storage
                loss_per_task[task_id]+=(torch.sum(
                   selected_mask * tokens_loss).item()
                )
                tokens_per_task[task_id]+= selected_mask.sum().item()

                
    # Compute final values
    results = {}
    final_mean = []
    for task_id, task in enumerate(names):
        # Get mean per task
        # 1. Check task tokens
        task_tokens = tokens_per_task[task_id].item()
        if task_tokens == 0:
            results[task] = 0.0
            continue
        # 2. Get loss
        task_loss = loss_per_task[task_id].item()
        # Populate dict -> mean ce per task
        results[task] = task_loss/task_tokens 
        # 3. Final mean contribution
        final_mean.append(results[task])

    # Include the final mean
    results["mean"] = torch.mean(torch.tensor(final_mean)).item()

    return results

