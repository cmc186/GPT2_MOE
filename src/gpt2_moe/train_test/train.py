# Basic Imports
import os
import sys
import json
import torch
import argparse
from pathlib import Path

# File system imports
from gpt2_moe.model import GPTMOE
from gpt2_moe.train_test.scorers import ScorerSelector
from gpt2_moe.train_test.utils import Utils
from gpt2_moe.train_test.configs import ExperimentConfig
from gpt2_moe.train_test.losses import LossSelector

# Get 2nd parent of the current file and build from there
parent_path = Path(__file__).parents[1].resolve()
BASE_PATH = os.path.join(parent_path, "checkpoints")
GPT2_NAME = 'gpt2-small-124M.pth'


# Define the parser
parser = argparse.ArgumentParser()
# Add the arguments
parser.add_argument('--exp', help = 'Select experiment name (E1,E2,E3)', default = 'E1', choices = ['E1', 'E2', 'E3'])
parser.add_argument('--dev', help = 'Select default device', default = 'cuda', choices = ['cuda', 'cpu'])
parser.add_argument('--name', help = 'Name of the config yaml', default = 'base.yaml')
parser.add_argument('--seed', help = 'Select seed', default = 42)

# Retrieve the arguments
args = parser.parse_args()
seed = int(args.seed)
exp = args.exp
dev = args.dev
name = args.name

# Info printing
print(f'(args) Selected seed ({seed})')
print(f'(args) Selected experiment ({exp})')
print(f'(args) Selected device ({dev})')
print(f'(args) Selected yaml ({name})')

# Initialize configurator
configurator = ExperimentConfig(name = name)


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
    model_storing_path = path + '.pth'
    data_storing_path = path + '_history.json'

    # 1. FREEZING AND ROUTING
    Utils.freeze_selective(model = model, groups = stage['groups'])
    model.set_forced_expert(stage['expert'])

    # 2. OPTIMIZER, LOSS AND SCHEDULER
    # 2.1 Pass only the trainable parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr = train_params['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = train_params['epochs'])
    # 2.2 Initialize the loss
    loss_fn = LossSelector.loss_selector(stage['loss'],  {'source': model.moe_layers, "params":stage['loss_params']} if stage['loss'] == 'balanced_loss' 
                                         else {})
    v_loss_fn = LossSelector.loss_selector('ce', {})


    # 3. TRAINING
    history = []
    best_val_score = float('-inf')

    for epoch in range(train_params['epochs']):
        # Set train mode
        model.train()
        # Training data
        train_loss = 0
        clipped_batches = 0
        epoch_grad_norm = 0
        # Iterate per batch
        for batch in train_loader:
            # Retrieve data 
            inputs = batch['input_ids'].to(device)
            targets = batch['target_ids'].to(device)
            #print('Aspect of inputs:', inputs.shape)
            #print('Aspect of target:', targets.shape)
            # Feed the model (B, N, V)
            logits = model(inputs)
            # Loss (B, N, V) -> (B*N, V)
            #print('Aspect of the logits:', logits.shape)
            loss = loss_fn(logits, targets)
            # Compute loss
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                                            parameters = trainable,
                                            max_norm  = 1.0)

            #print(grad_norm if grad_norm>=1.0 else '')
            # Register statistics
            epoch_grad_norm += grad_norm.item()
            clipped_batches+= grad_norm.item()>1
            # Optimizer step
            optimizer.step()
            # Update loss
            train_loss += loss.item()

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Register clipping statistic
        mean_grad_norm = epoch_grad_norm/len(train_loader)
        clip_ratio = clipped_batches/len(train_loader)
        #print(f"Average Grad Norm:{mean_grad_norm}"f" Clipping ratio beyond batches:{clip_ratio}")
        # Validation data
        model.eval()
        val_loss = 0
        val_score = 0
        with torch.no_grad():
            # Feed val batches to the model
            for batch in val_loader:
                # Retrieve data
                inputs = batch['input_ids'].to(device)
                targets = batch['target_ids'].to(device)
                # Get output (B, N, V)
                logits = model(inputs)
                # Compute loss
                val_loss += v_loss_fn(logits, targets).item()

            # Call specific metrics
            metrics = ScorerSelector().run(
                train_params.get('scorers', []),
                model, 
                val_loader,
                device = dev
            )


        # Mean val, loss per batch
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

    
        # Get validation score: exact_match based
        val_score = metrics["exact_match"]["mean"]

        # Register values
        history.append({
            'epoch': epoch, 
            'train_loss': train_loss, 
            'val_loss': val_loss,
            'lr_evo':current_lr,
            "grad_norm":mean_grad_norm,
            "clip_ratio":clip_ratio,
            **metrics
            })
        
        print(f" ·({stage['name']}) epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f} :{val_score:.4f}")

        # 4. STORING -> validation score triggered
        if val_score > best_val_score:
            best_val_score = val_score
            # Save the model
            torch.save(model.state_dict(), model_storing_path)

    # Restore routing 
    model.set_forced_expert(None)
    # Store the loss history
    with open(data_storing_path, 'w') as f:
        json.dump(history, f, indent = 2)


def run(seed:int = seed, device:str = dev)->GPTMOE:
    """
        Function aimed to run a whole experiment

        Params:
            seed(int):
                Seed for init, split and shuffle
            device(str):
                 Switch between cuda or cpu
                Default value is 'cpu' 

        Returns:
            model(GPTMOE):
                The model after the last stage
    """

    # Get the device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    # Seeding the model
    Utils.seed_setter(seed)


    # Retrieve model params and experiments
    model_params = configurator.get_model_params()
    hyperparams = configurator.get_train()
    # Debug prints
    print('DEBUG PRINTS')
    print('·The params:', model_params)
    print('·The hp:', hyperparams)
    print('·The init mode:', configurator.init_mode)
    print('·The load modality:', configurator.load_modality)
    print('·The name:', configurator.name)

    for i, stage in enumerate(configurator):
        print(f'(·Stage{i}):{stage}')

    # 1. BUILD AND INITIALIZE
    model = GPTMOE(configurator.get_model_params())

    # Load weights if required 
    if configurator.init_mode == 'gpt2':
        # Switch loading mode 
        Utils.load_weights(model = model, name = GPT2_NAME, base_path = BASE_PATH, blocks = model_params['n_layers'],
                            load_mod = configurator.load_modality, upcycling = configurator.upcycling)
    # Move model to the selected device
    model.to(device)

    # 2. ENSURE STORING PATH (gpt2_moe/checkpoints/name/seed)
    name = configurator.name
    run_path = os.path.join(BASE_PATH, name, f'seed_{seed}')
    Utils.ensure_path(run_path)

    # 3. STAGES PHASE
    for stage in configurator:
        # Define the path for the stage
        path = os.path.join(run_path, f"{stage['name']}")
        # Check existance
        if not os.path.exists(path + '.pth'):
            print(f"\n(run) STAGE {stage['name']}")
            # Get DataLoaders: in case != expert then train with everything
            loaders = Utils.get_loaders(tasks = stage['tasks'] or configurator.TASKS, batch_size = hyperparams['bs'],
                                        context_length = model_params['context_length'], seed = seed
                                    )
            # Call training
            train_stage(model = model, stage = stage, train_params = hyperparams, loaders = loaders, path = path, device = device)

        # Next stage starts from the best weights of this one
        model.load_state_dict(torch.load(path + '.pth', weights_only = True))

    return model


if __name__=='__main__':
    run()