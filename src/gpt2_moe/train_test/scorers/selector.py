# Basic Import 
import torch
import torch.nn as nn

# Conduct relative imports 
from .jenser_divergence import JensenDivergence

class LossSelector():

    def __init__(self):
        pass


    @staticmethod
    def scorer_selector(name:str, params:dict)->nn.Module:

        # Preprocess input 
        name = name.lower()

        # Return a scorer
        match name:
            case 'jsd':
                return JensenDivergence()




