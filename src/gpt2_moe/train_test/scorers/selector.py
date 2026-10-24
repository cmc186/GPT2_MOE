# Conduct relative imports
from .token_accuracy import token_accuracy
from .exact_match import exact_match
from .routing_divergence import routing_divergence

class ScorerSelector():

    # Name -> function(model, loader, device, **kwargs) -> dict
    SCORERS = {
        'token_acc': token_accuracy,
        'exact_match': exact_match,
        'routing_jsd': routing_divergence,
    }

    @staticmethod
    def run(names:list[str], model, loader, **kwargs)->dict:
        """
            Function aimed to run every requested scorer

            Params:
                names(list[str]):
                    The scorer names
                model(GPTMOE):
                    The model to evaluate
                loader(DataLoader):
                    Loader over a TaskDataset
                kwargs:
                    Extra params, each scorer takes the ones it needs

            Returns:
                results(dict):
                    {scorer_name: scorer_result}
        """
        # Set model to eval in case we forgot it
        model.eval()
        results = {}
        # Forward different scorers
        for name in names:
            if name not in ScorerSelector.SCORERS:
                continue
            # Get result
            results[name] = ScorerSelector.SCORERS[name](model, loader, **kwargs)
        return results
