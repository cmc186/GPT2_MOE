# Basic Import
import torch
import torch.nn as nn

# Conduct relative imports
from .balanced_loss import BalancedLoss
from .ce_loss import CrossEntropyLossWrapper

class LossSelector():

    def __init__(self):
        pass


    @staticmethod
    def loss_selector(name:str, params:dict)->nn.Module:

        # Preprocess input
        name = name.lower()

        # Return an uninitialized loss
        match name:

            case 'ce':
                return CrossEntropyLossWrapper(**params)

            case 'balanced_loss':
                return BalancedLoss(**params)




