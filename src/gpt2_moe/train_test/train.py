# Basic Imports
import os
import sys
import json
import torch

# File system imports
from gpt2_moe.model import GPTMOE
from gpt2_moe.train_test.utils import Utils
from gpt2_moe.train_test.losses import LossSelector
from gpt2_moe.train_test.config import ATTR, TRAIN, TASKS, EXPERIMENTS

# Paths (launch from the repo root, as data/)
BASE_PATH = 'src/gpt2_moe/checkpoints'
GPT2_NAME = 'gpt2-small-124M.pth'


def train_stage(model:GPTMOE, stage:dict, train_params:dict, loaders:tuple, path:str, device:str)->None:
    """
        Function aimed to train a single stage

        Params:
            model(GPTMOE):
                The model we are dealing with
            stage(dict):
                The stage config (groups, expert, tasks, loss)
            train_params(dict):
                epochs, bs and lr
            loaders(tuple):
                train, val and test loaders
            path(str):
                Where the best weights are stored
            device(str):
                cuda or cpu
    """
    train_loader, val_loader, _ = loaders

    # 1. FREEZING AND ROUTING
    Utils.freeze_selective(model = model, groups = stage['groups'])
    model.set_forced_expert(stage['expert'])

    # 2. OPTIMIZER AND LOSS
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr = train_params['lr'])
    loss_params = {'moe_layers': model.moe_layers()} if stage['loss'] == 'own' else {}
    loss_fn = LossSelector.loss_selector(stage['loss'], loss_params)

    # 3. TRAINING
    history = []
    best_val_loss = float('inf')

    for epoch in range(train_params['epochs']):

        # Training data
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
            # Feed
            logits = model(inputs)
            # Loss (B, N, V) -> (B*N, V)
            loss = loss_fn(logits.flatten(0, 1), targets.flatten())
            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm = 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation data
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input_ids'].to(device)
                targets = batch['target_ids'].to(device)
                logits = model(inputs)
                val_loss += loss_fn(logits.flatten(0, 1), targets.flatten()).item()

        # Mean per batch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"·({stage['name']}) epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

        # 4. STORING -> only when validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), path)

    # Restore routing and store the loss history
    model.set_forced_expert(None)
    with open(path.replace('.pth', '_history.json'), 'w') as f:
        json.dump(history, f, indent = 2)


def run(name:str, seed:int = 42, device:str = None)->GPTMOE:
    """
        Function aimed to run a whole experiment

        Params:
            name(str):
                Experiment name (E1, E2, E3)
            seed(int):
                Seed for init, split and shuffle
            device(str):
                cuda or cpu. None -> cuda if available

        Returns:
            model(GPTMOE):
                The model after the last stage
    """
    experiment = EXPERIMENTS[name]
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Seed before building -> router and random init are reproducible
    torch.manual_seed(seed)

    # 1. BUILD AND INITIALIZE
    model = GPTMOE(ATTR)
    if experiment['init'] == 'gpt2':
        Utils.load_weights(model = model, name = GPT2_NAME, base_path = BASE_PATH, blocks = ATTR['n_layers'])
    model.to(device)

    # Folder for this run 
    run_path = os.path.join(BASE_PATH, name, f'seed_{seed}')
    Utils.ensure_path(run_path)

    # 2. STAGES
    for stage in experiment['stages']:
        path = os.path.join(run_path, f"{stage['name']}.pth")

        # Stage already trained -> skip
        if not os.path.exists(path):
            print(f"\n(run) STAGE {stage['name']}")
            # Stage values override the common ones
            train_params = {**TRAIN, **stage.get('train', {})}
            loaders = Utils.get_loaders(tasks = stage['tasks'] or TASKS, batch_size = train_params['bs'],
                                        context_length = ATTR['context_length'], seed = seed)
            train_stage(model, stage, train_params, loaders, path, device)

        # Next stage starts from the best weights of this one
        model.load_state_dict(torch.load(path, map_location = device, weights_only = True))

    return model

