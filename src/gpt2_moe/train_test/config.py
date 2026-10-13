# Tasks in fixed order -> expert i is trained on TASKS[i]
TASKS = ["add_sub", "mul_div", "alphabetic", "spell", "parity", "first_last"]

# Model attributes -> same structure for the three experiments
ATTR = {
    "vocab_size": 50257,
    "context_length": 1024,
    "n_layers": 12,
    "emb_dim": 768,
    "dropout": 0.1,
    "ffn_hidden_dim": None,
    "moe_mask": {
        "schedule": "alternate",
        "offset": 2
    },
    "attention_related": {
        "num_heads": 12,
        "dropout": 0.1,
        "qkv_bias": True
    },
    "moe_related": {
        "experts": len(TASKS),
        "k_routing": 1,
        "hidden_dim": None,
    },
}

# Training params shared by every stage
TRAIN = {"epochs": 10, "bs": 8, "lr": 3e-4}

# One stage per expert: expert i only sees task i, router skipped
EXPERT_STAGES = [
    {"name": f"expert_{i}", "groups": ["experts"], "expert": i, "tasks": [task], "loss": "ce"}
    for i, task in enumerate(TASKS)
]

# Experiments -> init mode + ordered list of stages
EXPERIMENTS = {
    # 1- Experts one by one, then router + FF
    "E1": {
        "init": "gpt2",
        "stages": EXPERT_STAGES + [
            {"name": "router_ffn", "groups": ["router", "ffn"], "expert": None, "tasks": None, "loss": "own"},
        ],
    },
    # 2- Experts one by one, then router only (FF always frozen)
    "E2": {
        "init": "gpt2",
        "stages": EXPERT_STAGES + [
            {"name": "router", "groups": ["router"], "expert": None, "tasks": None, "loss": "own"},
        ],
    },
    # 3- Everything at once from random weights
    "E3": {
        "init": "random",
        "stages": [
            {"name": "joint", "groups": ["base", "ffn", "experts", "router"], "expert": None, "tasks": None, "loss": "own"},
        ],
    },
}
