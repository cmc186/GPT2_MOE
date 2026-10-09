from gpt2_moe.data.tokenizer import Tokenizer
from gpt2_moe.data.dataset import TaskDataset
from gpt2_moe.data.dataloader import DataLoader

csvs = [
    "data/add_sub.csv",
    "data/mul_div.csv",
    "data/alphabetic.csv",
    "data/spell.csv",
    "data/parity.csv",
    "data/first_last.csv",
]
tok = Tokenizer(embedding_dim=768, max_seq_len=64)


print("\nTOKENIZER")
text = "add_sub: 37 + 12 = 49"
ids = tok.encode(text)
print("texto  :", text)
print("ids    :", ids)
print("tokens :", [tok.decode([i]) for i in ids])


print("\nDATASET")
ds = TaskDataset(tok, 64, datasets=csvs, split="train")
ex = ds[0]
print("ejemplos en train:", len(ds))
print("tarea  :", ds.task_names[ex["task"]])
print("texto  :", tok.decode(ex["input_ids"].tolist()))
print("\npos | input           | target")
for i, (a, b) in enumerate(zip(ex["input_ids"].tolist(), ex["target_ids"].tolist())):
    target = "-100 (no cuenta)" if b == -100 else repr(tok.decode([b]))
    print(f"{i:>3} | {tok.decode([a])!r:<15} | {target}")


print("\nDATALOADER")
data = DataLoader(tok, context_lenght=64, batch_size=4, datasets=csvs)
batch = next(iter(data.train_loader))
print("max_lenght:", data.max_lenght)
print("input_ids :", tuple(batch["input_ids"].shape), " -> (ejemplos, tokens)")
print("target_ids:", tuple(batch["target_ids"].shape))
print("task      :", tuple(batch["task"].shape))

for i in range(len(batch["task"])):
    ids = batch["input_ids"][i].tolist()
    real = [t for t in ids if t != data.pad_id]
    answer = [t for t in batch["target_ids"][i].tolist() if t != -100]
    print(f"\nejemplo {i} ({ds.task_names[batch['task'][i]]})")
    print("  texto    :", tok.decode(real))
    print("  respuesta:", repr(tok.decode(answer)))
    print("  tokens   :", len(real), "reales +", len(ids) - len(real), "de padding")