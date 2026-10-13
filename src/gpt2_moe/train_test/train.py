# Basic Imports
import os
import sys
import json
import torch

# File system imports
from gpt2_moe.model import GPTMOE
from gpt2_moe.train_test.utils import Utils
from gpt2_moe.train_test.losses import LossSelector
from gpt2_moe.train_test.config import MODEL_ATTR, TRAIN, TASKS, EXPERIMENTS

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
        Returns:
            None:
                No value is returned
    """
    # Extract loaders
    train_loader, val_loader, _ = loaders
    # Define storing paths
    model_storing_path = os.path.join(path, '.pth')
    data_storing_path = os.path.join(path, '_history.json')

    # 1. FREEZING AND ROUTING
    Utils.freeze_selective(model = model, groups = stage['groups'])
    model.set_forced_expert(stage['expert'])

    # 2. OPTIMIZER AND LOSS
    # 2.1 Pass only the trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr = train_params['lr'])
    # 2.2 Initialize the loss
    loss_fn = LossSelector.loss_selector(stage['loss'],  {'moe_layers': model.moe_layers()} if stage['loss'] == 'balanced_loss' else {})


    # 3. TRAINING
    history = []
    model.train()
    best_val_loss = float('inf')

    for epoch in range(train_params['epochs']):
        # Training data
        train_loss = 0
        # Iterate per batch
        for batch in train_loader:
            # Retrieve data 
            inputs = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
            # Feed the model (B, N, V)
            logits = model(inputs)
            # Loss (B, N, V) -> (B*N, V)
            loss = loss_fn(logits.flatten(0, 1), targets.flatten())
            # Compute loss
            optimizer.zero_grad()
            loss.backward()
            # Optimizer step
            optimizer.step()
            # Update loss
            train_loss += loss.item()

        # Validation data
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # Feed val batches to the model
            for batch in val_loader:
                # Retrieve data
                inputs = batch['input_ids'].to(device)
                targets = batch['target_ids'].to(device)
                # Get output (B, N, V)
                logits = model(inputs)
                # Compute loss
                val_loss += loss_fn(logits.flatten(0, 1), targets.flatten()).item()

        # Mean loss per batch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        # Store values
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"·({stage['name']}) epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

        # 4. STORING -> validation triggered
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            torch.save(model.state_dict(), model_storing_path)

    # Restore routing 
    model.set_forced_expert(None)
    # Store the loss history
    with open(data_storing_path, 'w') as f:
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
    # Retrieve the experiment (E1, E2, E3)
    experiment = EXPERIMENTS[name]
    # Select the device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Seed before building -> router and random init are reproducible
    Utils.seed_setter(seed)

    # 1. BUILD AND INITIALIZE: config from config.py
    model = GPTMOE(MODEL_ATTR)
    # Load GPT2 base weights
    if experiment['init'] == 'gpt2':
        # Switch loading mode 
        Utils.load_weights(model = model, name = GPT2_NAME, base_path = BASE_PATH, blocks = MODEL_ATTR['n_layers'], load_mod = experiment['load_modality'])
    # Move model to the selected device
    model.to(device)

    # 2. ENSURE STORING PATH (gpt2_moe/checkpoints)
    run_path = os.path.join(BASE_PATH, name, f'seed_{seed}')
    Utils.ensure_path(run_path)

    # 3. STAGES PHASE
    for stage in experiment['stages']:
        # Define the path for the stage
        path = os.path.join(run_path, f"{stage['name']}")

        # Stage already trained -> skip
        if not os.path.exists(path):
            print(f"\n(run) STAGE {stage['name']}")
            # Stage values override the common ones
            train_params = {**TRAIN, **stage.get('train', {})}
            # Get DataLoaders
            loaders = Utils.get_loaders(tasks = stage['tasks'] or TASKS, batch_size = train_params['bs'],
                                        context_length = MODEL_ATTR['context_length'], seed = seed
                                    )

            # Call training
            train_stage(model, stage, train_params, loaders, path, device)

        # Next stage starts from the best weights of this one
        model.load_state_dict(torch.load(path + '.pth', weights_only = True))

    return model

