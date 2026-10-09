# Basic Import 
import torch
import torch.nn as nn

# Conduct relative imports 
from .own_scorer import OwnScorer

class LossSelector():

    def __init__(self):
        pass


    @staticmethod
    def scorer_selector(name:str, params:dict)->nn.Module:

        # Preprocess input 
        name = name.lower()

        # Return a scorer
        match name:

            case 'own':
                return OwnScorers




