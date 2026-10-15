# Basic Imports
import os
import torch
import requests
import torch.nn as nn

# File system imports
from gpt2_moe.model import GPTMOE
from gpt2_moe.data.dataset import DATA_DIR
from gpt2_moe.data.tokenizer import Tokenizer
from gpt2_moe.data.dataloader import DataLoader


class Utils:
    """
        Class containing all the utils that
        could be used during the train-test
        process

    """

    @staticmethod
    def download_gpt2(name: str, model_dir:str)->torch.Tensor:
        """
            Function aimed to load GPT2 weights

            Params:
                name(str):
                    The name of the model
                model_dir(str):
                    Base dir for model storage

            Returns:
                weights (torch.Tensor):
                    Tensor representing weights

        """
        # Check path existance
        full_path = os.path.join(model_dir, name)
        if not os.path.exists(full_path):
            print(f'·Currently downloading ({name})')

            # Build the base folder
            os.makedirs(model_dir, exist_ok = True)
            # Define the url
            url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{name}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(full_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded to {full_path}")

    @staticmethod
    def assign(left:torch.Tensor, right:torch.Tensor):
        """
            Function aimed to assign weights
        """
        if left.shape!=right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

        # Assign weights
        return nn.Parameter(right)
    
    @staticmethod
    def load_weights(model:GPTMOE, name:str, base_path:str, blocks:int, load_mod:bool)->None:
        """
            Function aimed to load weights into a given
            model. GPT skeleton architecture is expected

            Params:
                model (GPTMOE):
                    The model to initialize

                name(str):
                    The name of the model to load

                base_path (dict):
                    The dictionary containing weights

                blocks (int):
                    The amount of tf blocks to initialize

                load_mod (bool):
                    Decided whether FFN are loaded or not

            Returns:
                None:
                    No value is returned

        """

        # Build the path and get the params
        specific_folder = 'base_weights'
        folder_path = os.path.join(base_path, specific_folder)
        path = os.path.join(folder_path, name)

        # Check if params are there -> ensure
        if not os.path.exists(path):
            Utils.download_gpt2(name = name, model_dir = folder_path)

        # Retrive params
        params = torch.load(path, weights_only=True)
        print(f'\n(general_load) LOADING BASE MODEL WEIGHTS')
        # Acquire embeddings
        model.tok_emb.weight = Utils.assign(model.tok_emb.weight, params["tok_emb.weight"])
        model.pos_emb.weight = Utils.assign(model.pos_emb.weight, params["pos_emb.weight"])

        # Output norm and language head
        model.final_norm.scale = Utils.assign(model.final_norm.scale, params["final_norm.scale"])
        model.final_norm.shift = Utils.assign(model.final_norm.shift, params["final_norm.shift"])
        model.out_head.weight = Utils.assign(model.out_head.weight, params["out_head.weight"])

        # Initialize tf blocks
        for i in range(blocks):
            # MHA BLOCK
            # Initialize query
            model.trf_blocks[i].att.W_query.weight = Utils.assign(model.trf_blocks[i].att.W_query.weight,
                                                            params[f"trf_blocks.{i}.att.W_query.weight"])

            model.trf_blocks[i].att.W_query.bias = Utils.assign(model.trf_blocks[i].att.W_query.bias,
                                                                params[f"trf_blocks.{i}.att.W_query.bias"])
            # Initialize key
            model.trf_blocks[i].att.W_key.weight = Utils.assign( model.trf_blocks[i].att.W_key.weight,
                                                        params[f"trf_blocks.{i}.att.W_key.weight"])

            model.trf_blocks[i].att.W_key.bias = Utils.assign( model.trf_blocks[i].att.W_key.bias,
                                                                params[f"trf_blocks.{i}.att.W_key.bias"])
            # Initialize value
            model.trf_blocks[i].att.W_value.weight = Utils.assign( model.trf_blocks[i].att.W_value.weight,
                                                            params[f"trf_blocks.{i}.att.W_value.weight"])

            model.trf_blocks[i].att.W_value.bias = Utils.assign( model.trf_blocks[i].att.W_value.bias,
                                                                params[f"trf_blocks.{i}.att.W_value.bias"])
            # Initialize out proj
            model.trf_blocks[i].att.out_proj.weight = Utils.assign( model.trf_blocks[i].att.out_proj.weight, params[f"trf_blocks.{i}.att.out_proj.weight"])
            model.trf_blocks[i].att.out_proj.bias = Utils.assign( model.trf_blocks[i].att.out_proj.bias, params[f"trf_blocks.{i}.att.out_proj.bias"])


            # Initialize normalizations (within tf blocks)
            model.trf_blocks[i].norm1.scale = Utils.assign(model.trf_blocks[i].norm1.scale, params[f"trf_blocks.{i}.norm1.scale"])
            model.trf_blocks[i].norm1.shift = Utils.assign(model.trf_blocks[i].norm1.shift, params[f"trf_blocks.{i}.norm1.shift"])
            model.trf_blocks[i].norm2.scale = Utils.assign(model.trf_blocks[i].norm2.scale, params[f"trf_blocks.{i}.norm2.scale"])
            model.trf_blocks[i].norm2.shift = Utils.assign(model.trf_blocks[i].norm2.shift, params[f"trf_blocks.{i}.norm2.shift"])

            # Load_mod is true when freezed FFN are required
            if load_mod:
                # FF block -> only want to load it when E2: FF always frozen
                ff = model.trf_blocks[i].ff
                # Load the ffn layers
                ff.ffn[0].weight = Utils.assign(ff.ffn[0].weight, params[f"trf_blocks.{i}.ff.layers.0.weight"])
                ff.ffn[0].bias = Utils.assign(ff.ffn[0].bias, params[f"trf_blocks.{i}.ff.layers.0.bias"])
                ff.ffn[2].weight = Utils.assign(ff.ffn[2].weight, params[f"trf_blocks.{i}.ff.layers.2.weight"])
                ff.ffn[2].bias = Utils.assign(ff.ffn[2].bias, params[f"trf_blocks.{i}.ff.layers.2.bias"])

            '''
                We do not want experts Fine Tuning
                if model.moe_mask[i]:
                    for expert in ff.experts:
                        # clone -> each expert gets its own copy
                        expert.l1.weight = Utils.assign(expert.l1.weight, w1.clone())
                        expert.l1.bias = Utils.assign(expert.l1.bias, b1.clone())
                        expert.l2.weight = Utils.assign(expert.l2.weight, w2.clone())
                        expert.l2.bias = Utils.assign(expert.l2.bias, b2.clone())
            '''
           
    '''
    @staticmethod
    def load_block(model:GPTMOE, base_path:str, block_idx:int, version:str)->None:
        """
            Function aimed to load the a single block
            into the GPT2MOE architecture. This function won't
            be accessed until the k experts have been individually
            trained.

            Expected storing format for the experts and ffn the following:
            regimes/
                    1/
                        block_index/
                            expert_0.pt
                            expert_1.pt
                            |
                            |
                            expert_n.pt

                        block_index/
                            ffn.pt

            Params:
                model(GPTMOE):
                    The model whose experts we want to set up
                    Default value is None
                base_path (str):
                    The base path where experts live
                block_idx(int):
                    The block where expert is being loaded

                version(str):
                    The version name of the current model
        """

        # Put together path and block
        path = os.path.join(base_path, version, str(block_idx))
        #print('(single_load) Aspect of the path:', path)


        # Retrieve the ff block
        block = model.trf_blocks[block_idx].ff

        # Check if we are facing moe or ff
        selected_experts = model.moe_mask[block_idx]

        # Experts selected
        if selected_experts:
            #print('DEALING WITH EXPERTS')
            # Define expert chain
            name = 'expert_{index}.pth'
            n_experts = len(block.experts)
            # Iterate over the available experts
            for i in range(n_experts):
                print(f' ->(expert_load) Loading expert ({i})')
                # Get to the expert i
                full_path = os.path.join(path, name.format(index = i))
                print('     ·Aspect of the full path:', full_path)
                # Retrieve the state dict of the expert
                expert_state_dict = torch.load(full_path, weights_only = True)
                # Load the state dict into a selected model's module
                block.experts[i].load_state_dict(expert_state_dict)


        # FFN selected
        else:
            #print('DEALING WITH FFN')
            # Define expert chain
            name = 'ffn.pth'
            # Get the full path to the ffn
            full_path = os.path.join(path, name)
            print(' ->(ffn)Aspect of the full path:', full_path)
            # Retrieve the state dict of the ffn
            ffn_state_dict = torch.load(full_path, weights_only = True)
            # Load the state dict into a selected model's module
            block.ffn.load_state_dict(ffn_state_dict)
    @staticmethod
    def load_all(model:GPTMOE, base_path:str, version:str)->None:
        """
            Function aimed to load all the additional blocks
            added to the base GPT2 skeleton. This function won't
            be accessed until the k experts have been individually
            trained.

            Expected storing format for the experts and ffn is the following:
            regimes/
                    1/
                        block_index/
                            expert_0.pt
                            expert_1.pt
                            |
                            |
                            expert_n.pt

                        block_index/
                            ffn.pt

            Params:
                model(GPTMOE):
                    The model whose experts we want to set up
                base_path (str):
                    The base path where experts live

                version(str):
                    The version of the model we are dealing with

        """

        # Info show
        print(f'\n(load_all)LOADING AVAILABLE BLOCKS')
        # Iterate over all blocks
        for i in range(len(model.trf_blocks)):
            # Loading each block
            print(f'·(blocks_loader)Loading block ({i})')
            Utils.load_block(model, base_path = base_path, block_idx = i, version = version)
    '''
    @staticmethod
    def freeze_selective(model:GPTMOE, groups:list[str])->None:
        """
            Function aimed to selectively freeze
            model parameters. Only the given groups
            are left trainable.

            Groups:
                base: embeddings, attention, norms and head
                ffn: FF layers (non MoE blocks)
                experts: experts of MoE blocks
                router: gate of MoE blocks

            Params:
                model(GPTMOE):
                    The model to work with
                groups(list[str]):
                    The  defined groups to train
            Returns:
                None:
                    No value is returned
        """

        # Freeze the whole model
        model.requires_grad_(False)

        # Base model
        if 'base' in groups:
            # Unfreeze the model
            model.requires_grad_(True)
            # Setting trf blocks to false waiting for other triggers
            for block in model.trf_blocks:
                block.ff.requires_grad_(False)

        # Heat the requested ff parts
        for block in model.trf_blocks:
            # Heating MoE
            if block.use_moe:
                if 'experts' in groups:
                    block.ff.experts.requires_grad_(True)
                if 'router' in groups:
                    block.ff.gate.requires_grad_(True)
            # Heating the FFN
            elif 'ffn' in groups:
                block.ff.requires_grad_(True)
    '''
    @staticmethod
    def params_storage(model:GPTMOE, base_path:str, version:str, train_mode:bool, id:int)->None:
        """
            Function aimed to store data based on
            the training mode that is being conducted

            Train modes are the following:

            1) experts(True): only experts are stored
            2) routers(False): the whole model is stored

            Params:
                model(GPTMOE):
                    The model we are currently using
                base_path(str):
                    The base path to store data at
                version(str):
                    The version of the model we are dealing with
                train_mode(bool):
                    The train mode we are following
                id(int):
                    The current id

        """

        # Info printing
        print(f'\n(params_storage) STORING PARAMS')
        print(f'·Conducting ({'experts|ff' if train_mode else 'whole model'}) storing')
        # Build the base path for storing ('gpt2_moe/checkpoints' + version)
        base_path = os.path.join(base_path, version)

        # Switch train mode
        if train_mode:
            # Save the experts and ffn only
            for i, block in enumerate(model.trf_blocks):
                # Define the full_path
                full_path = os.path.join(base_path, str(i))
                # Check mask
                if model.moe_mask[i]:
                    # We are training one expert at a time
                    name = 'expert_' + str(id) + '.pth'
                    # Experts is always index 0: one training at a time
                    state_dict = block.ff.experts[0].state_dict()
                else:
                    # Define ffn name and state dict
                    name = 'ffn_' + str(id) + '.pth'
                    state_dict = block.ff.ffn.state_dict()

                # Store transparently
                torch.save(state_dict, os.path.join(full_path, name))
        else:
            # Define path
            path = os.path.join(base_path, 'model')
            # Get model's available name
            name = 'model_' + str(id) +  '.pth'

            path_name = os.path.join(path, name)
            # Store the whole model with the trained router
            torch.save(name, path_name)
    '''
    @staticmethod
    def ensure_path(path:str)->None:
        """
            Function aimed to ensure path existance
            before storing

            Params:
                path(str):
                    The path to check
            Returns:
                None:
                    No value is returned

        """

        # Ensure path existance
        os.makedirs(path, exist_ok = True)

    @staticmethod
    def get_next_name(path:str, key:str, ext:str = '')->str:
        """
            Function aimed to return the next available
            name given a base namekey. Names will be
            built using the format name_(next_integer)

            Params:
                path(str):
                    The base path to look at

                key (str):
                    The base name

            Returns:
                name(str):
                    The first available name

        """
        # Ensure path exisance
        Utils.ensure_path(path)

        # Get dir entries
        entries = [entry.name for entry in os.scandir(path)]
        last_idx = -1

        # Check entries value
        if entries:
            # Get last valid idx
            indices = [
                int(name.split('.')[0].split('_')[-1]) for name in entries
            ]
            # Sort indices
            last_idx = sorted(indices)[-1]
            print('(next_name) Aspect of the last used idx:', last_idx)

        # Create the new idx -> if dir empty the new index will be 0
        new_idx = last_idx + 1
        print('(next_name) Aspect of the new idx:', new_idx)

        # Build the new name
        new_name = key + '_' + str(new_idx) + ext
        print('(next_name) Aspect of the new name:', new_name)
        return new_name

    @staticmethod
    def get_loaders(tasks:list[str], batch_size:int, context_length:int, seed:int = 42)->tuple:
        """
            Function aimed to build the train, val and test
            loaders using only the CSVs of the given tasks

            Params:
                tasks(list[str]):
                    The task names (names must match csvs)
                batch_size(int):
                    Examples per batch
                context_length(int):
                    Max length the model accepts
                seed(int):
                    Seed for split and shuffle
            Returns:
                loaders(tuple):
                    train, val and test loaders
        """
        # Build the tokenizer and the CSV paths
        tokenizer = Tokenizer(embedding_dim = 768, max_seq_len = context_length)
        datasets = [os.path.join(DATA_DIR, f'{task}.csv') for task in tasks]

        # Same split and padding logic as data/
        data = DataLoader(tokenizer, context_lenght = context_length, batch_size = batch_size,
                          seed = seed, datasets = datasets)
        return data.get_loaders()

    @staticmethod
    def seed_setter(seed:int)->None:
        """ 
            Function aimed to ensure experiment's 
            reproducibility

            Params:
                seed(int):
                    The selected seed

            Returns:
                None:
                    No value is returned
        """

        # Set the seed
        torch.manual_seed(seed)

