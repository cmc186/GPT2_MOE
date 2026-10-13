import torch
from gpt2_moe.model import GPTMOE
from gpt2_moe.train_test.utils import Utils

def attr_with(schedule):
    return {
        "vocab_size": 50257, "context_length": 1024, "n_layers": 12, "emb_dim": 768,
        "dropout": 0.0, "ffn_hidden_dim": None,
        "moe_mask": {"schedule": schedule, "offset": 2},
        "attention_related": {"num_heads": 12, "dropout": 0.0, "qkv_bias": True},
        "moe_related": {"experts": 6, "k_routing": 1, "hidden_dim": None},
    }

base_path = "gpt2_moe/checkpoints"   # lanzar desde src/
name = "gpt2-small-124M.pth"
x = torch.randint(0, 50257, (1, 16))

dense = GPTMOE(attr_with("none"))
Utils.load_weights(dense, name, base_path, blocks = 12)

moe = GPTMOE(attr_with("alternate"))
Utils.load_weights(moe, name, base_path, blocks = 12)
moe.set_forced_expert(3)

with torch.no_grad():
    assert torch.allclose(dense.eval()(x), moe.eval()(x), atol = 1e-5)

# Cada experto tiene su propia copia
e = moe.moe_layers()[0].experts
assert e[0].l1.weight.data_ptr() != e[1].l1.weight.data_ptr()

print("Bloque 2 OK")
