# Basic Imports
import torch

# Ignored positions in the targets (prompt and padding)
IGNORE_INDEX = -100


def per_task(correct:torch.Tensor, total:torch.Tensor, names:list[str])->dict:
    """
        Function aimed to turn per task counters
        into a {task: ratio} dict plus the macro mean

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
    results = {names[i]: (correct[i] / total[i]).item() for i in range(len(names)) if total[i] > 0}
    results['mean'] = sum(results.values()) / len(results) if results else 0.0
    return results


def split_example(input_ids:torch.Tensor, target_ids:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    """
        Function aimed to recover the prompt and the answer
        from a padded example

        Params:
            input_ids(torch.Tensor):
                Padded input ids (N,)
            target_ids(torch.Tensor):
                Padded target ids (N,) with the prompt masked

        Returns:
            prompt(torch.Tensor):
                Prompt ids, the last one predicts the first answer token
            answer(torch.Tensor):
                Answer ids without <|endoftext|>
    """
    # First non-masked target -> the last prompt token predicts it
    ans_pos = (target_ids != IGNORE_INDEX).nonzero()[0].item()
    prompt = input_ids[:ans_pos + 1]
    # Answer tokens + eos -> drop eos
    answer = target_ids[target_ids != IGNORE_INDEX][:-1]
    return prompt, answer


@torch.no_grad()
def greedy_answer(model, prompt:torch.Tensor, eos_id:int, max_new:int)->list[int]:
    """
        Function aimed to generate an answer with greedy
        decoding until <|endoftext|> or max_new tokens

        Params:
            model(GPTMOE):
                The model to generate with
            prompt(torch.Tensor):
                Prompt ids (P,) already on the device
            eos_id(int):
                Id of <|endoftext|>
            max_new(int):
                Max amount of generated tokens

        Returns:
            out(list[int]):
                Generated ids without <|endoftext|>
    """
    context_length = model.pos_emb.num_embeddings
    ids = prompt.unsqueeze(0)
    out = []
    for _ in range(max_new):
        # Logits of the last position (|V|,)
        next_id = model(ids[:, -context_length:])[0, -1].argmax()
        if next_id.item() == eos_id:
            break
        out.append(next_id.item())
        ids = torch.cat([ids, next_id.view(1, 1)], dim = 1)
    return out
