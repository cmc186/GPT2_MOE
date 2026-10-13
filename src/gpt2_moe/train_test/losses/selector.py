# Basic Import
import torch
import torch.nn as nn

# Conduct relative imports
from .own_loss import OwnLoss

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
                return nn.CrossEntropyLoss

            case 'own':
                return OwnLoss




