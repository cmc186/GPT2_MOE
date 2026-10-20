# Basic Imports
import torch

# Ignored positions in the targets (prompt and padding)
IGNORE_INDEX = -100


def per_task(correct:torch.Tensor, total:torch.Tensor, names:list[str])->dict:
    """
        Function aimed to turn per task counters
        into a {task: ratio} dict and provide a general
        interclass mean value

        Params:
            correct(torch.Tensor):
                Hits per task index (n_tasks,)
            total(torch.Tensor):
                Evaluated items per task index (n_tasks,)
            names(list[str]):
                Task names ordered by index

        Returns:
            results(dict):
                {task_name: ratio, ..., 'mean': macro mean}
    """
    # Build the dict: exclude unexistent classes: ordered experts training approach
    results = {names[i]: (correct[i] / total[i]).item() for i in range(len(names)) if total[i] > 0 }
    # Compute the general mean
    results['mean'] = sum(results.values()) / len(results)
    return results


def split_example(input_ids:torch.Tensor, target_ids:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    """
        Function aimed to recover the prompt and the answer
        from a padded example

        Params:
            input_ids(torch.Tensor):
                Padded input ids (N,)
            target_ids(torch.Tensor):
                Padded target ids (N,) including the prompt masked

        Returns:
            prompt(torch.Tensor):
                Prompt ids, the last one predicts the first answer token
            answer(torch.Tensor):
                Answer ids without <|endoftext|>
    """
    # First non-masked index from target
    ans_pos = torch.where(target_ids != IGNORE_INDEX)[0][0]
    # Answer is prompt +  tab (1)
    prompt = input_ids[:ans_pos + 1]
    # Get answer without padding (-100)
    answer = target_ids[target_ids != IGNORE_INDEX][:-1]
    return prompt, answer


def greedy_answer(model, prompt:torch.Tensor, eos_id:int, max_new:int)->list[int]:
    """
        Function aimed to generate an answer with greedy
        decoding until <|endoftext|> or max_new tokens 
        are reached

        Params:
            model(GPTMOE):
                The model to generate with
            prompt(torch.Tensor):
                Prompt ids (,) already on the device
            eos_id(int):
                Id of <|endoftext|>
            max_new(int):
                Max amount of generated tokens

        Returns:
            out(list[int]):
                Lis of generated ids without <|endoftext|>
    """
    # Get the context length
    context_length = model.pos_emb.num_embeddings
    # Modify ids shape (N, 1)
    ids = prompt.unsqueeze(0)
    output = []
    #print('Aspect of the ids:', ids, ids.shape)
    # Generative task
    for _ in range(max_new):
        # Logits of the last position (|V|,)
        next_id = torch.argmax(model(ids[:, -context_length:])[0, -1])
        # Check if answer is finished
        if next_id.item() == eos_id:
            break
        # Compose the output chain
        output.append(next_id.item())
        # Update the context window for the model
        ids = torch.cat([ids, next_id.view(1, 1)], dim = 1)

    return output
