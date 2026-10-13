"""
Comprobación de los bloques 1-6.

Lanzar desde la raíz del repo:
    PYTHONPATH=./src python check.py

No descarga nada ni toca checkpoints/ ni data/: los datos y checkpoints
de prueba se crean en una carpeta temporal que se borra al terminar.
"""
import os
import json
import shutil
import tempfile
import traceback
import torch
import torch.nn as nn

from gpt2_moe.model import GPTMOE
from gpt2_moe.model.model_utils import moe_mask
from gpt2_moe.train_test.utils import Utils
from gpt2_moe.train_test import config
import gpt2_moe.train_test.utils.utils as utils_mod
import gpt2_moe.train_test.train as train_mod

torch.manual_seed(0)
GPT2_NAME = 'gpt2-small-124M.pth'
RESULTS = []
WARNINGS = []


# ---------------------------------------------------------------- helpers

class Pending(Exception):
    """ Parte que no depende de nosotros (losses / scorers) """

class Skip(Exception):
    """ No se puede comprobar (falta un archivo) """

def section(title):
    """ Ejecuta la sección al definirla y guarda el resultado """
    def run(fn):
        try:
            fn()
            RESULTS.append((title, 'OK', ''))
        except Pending as e:
            RESULTS.append((title, 'PENDIENTE', str(e)))
        except Skip as e:
            RESULTS.append((title, 'SALTADO', str(e)))
        except Exception as e:
            traceback.print_exc()
            RESULTS.append((title, 'FALLA', f'{type(e).__name__}: {e}'))
    return run

def make_attr(schedule = 'alternate', n_layers = 2, emb_dim = 768, heads = 12, context = 1024, experts = 6):
    return {
        "vocab_size": 50257, "context_length": context, "n_layers": n_layers, "emb_dim": emb_dim,
        "dropout": 0.0, "ffn_hidden_dim": None,
        "moe_mask": {"schedule": schedule, "offset": 2},
        "attention_related": {"num_heads": heads, "dropout": 0.0, "qkv_bias": True},
        "moe_related": {"experts": experts, "k_routing": 1, "hidden_dim": None},
    }

def small_model(n_layers = 4):
    """ Modelo pequeño para pruebas rápidas en CPU """
    return GPTMOE(make_attr(n_layers = n_layers, emb_dim = 64, heads = 4))

def changed_keys(sd_a, sd_b):
    """ Claves del state_dict cuyo valor ha cambiado """
    return {k for k in sd_a if not torch.equal(sd_a[k], sd_b[k])}

def find_gpt2():
    """ Carpeta base donde está el checkpoint de GPT-2 (o None) """
    for base in ('src/gpt2_moe/checkpoints', 'gpt2_moe/checkpoints'):
        if os.path.exists(os.path.join(base, 'base_weights', GPT2_NAME)):
            return base
    return None


# Datos sintéticos de prueba: 2 tareas de 64 filas en una carpeta temporal
TMP = tempfile.mkdtemp(prefix = 'gpt2moe_check_')
DATA_TMP = os.path.join(TMP, 'data')
os.makedirs(DATA_TMP)
with open(os.path.join(DATA_TMP, 'task_a.csv'), 'w') as f:
    f.write('prompt,y\n' + ''.join(f'{a} + {b},{a + b}\n' for a in range(8) for b in range(8)))
with open(os.path.join(DATA_TMP, 'task_b.csv'), 'w') as f:
    f.write('prompt,y\n' + ''.join(f'{n},{n % 2}\n' for n in range(100, 164)))


# ---------------------------------------------------------------- bloque 1

@section('Bloque 1 · Modelo')
def _():
    model = small_model()
    x = torch.randint(0, 50257, (4, 32))

    # Forma y número de capas MoE / expertos
    logits = model(x)
    assert logits.shape == (4, 32, 50257), f'forma de salida {tuple(logits.shape)}'
    assert len(model.moe_layers()) == 2, 'alternate con offset 2 y 4 capas debería dar 2 capas MoE'

    # Estado del router
    for moe in model.moe_layers():
        assert len(moe.experts) == 6, f'hay {len(moe.experts)} expertos en vez de 6'
        assert moe.last_probs.shape == (128, 6), f'last_probs {tuple(moe.last_probs.shape)}'
        assert torch.allclose(moe.last_probs.sum(-1), torch.ones(128), atol = 1e-5), 'las probabilidades no suman 1'

    # Gradiente al router con k = 1
    nn.functional.cross_entropy(logits.flatten(0, 1), x.flatten()).backward()
    for moe in model.moe_layers():
        assert moe.gate.linear.weight.grad.abs().sum() > 0, 'el router no recibe gradiente con k = 1'

    # Experto forzado
    model.set_forced_expert(2)
    moe = model.moe_layers()[0]
    h = torch.randn(2, 5, 64)
    assert torch.equal(moe(h), moe.experts[2](h)), 'el experto forzado no coincide con experts[2]'
    assert moe.last_probs is None, 'con experto forzado last_probs debería ser None'
    model.set_forced_expert(None)

    # moe_mask rechaza configuraciones inválidas
    for bad in [dict(schedule = 'alternat'), dict(schedule = 'alternate', offset = 0),
                dict(schedule = []), dict(schedule = [0, 99])]:
        try:
            moe_mask(4, **bad)
        except ValueError:
            continue
        raise AssertionError(f'moe_mask no rechaza {bad}')


# ---------------------------------------------------------------- bloque 2

@section('Bloque 2 · Pesos GPT-2')
def _():
    base = find_gpt2()
    if base is None:
        raise Skip(f'no se encuentra {GPT2_NAME} (no se descarga desde aquí)')
    if base != train_mod.BASE_PATH:
        WARNINGS.append(f'El checkpoint de GPT-2 está en "{base}", pero train.py lo busca en '
                        f'"{train_mod.BASE_PATH}" y lo volvería a descargar. Muévelo allí.')

    x = torch.randint(0, 50257, (1, 16))

    # Mismo GPT-2 en un modelo sin MoE y en uno con MoE forzado
    dense = GPTMOE(make_attr('none'))
    Utils.load_weights(dense, GPT2_NAME, base, blocks = 2)
    moe = GPTMOE(make_attr('alternate'))
    Utils.load_weights(moe, GPT2_NAME, base, blocks = 2)
    moe.set_forced_expert(3)
    with torch.no_grad():
        diff = (dense.eval()(x) - moe.eval()(x)).abs().max().item()
    assert diff < 1e-4, f'el MoE forzado no da lo mismo que el denso (diferencia máx {diff:.2e})'

    # final_norm y out_head cargados
    params = torch.load(os.path.join(base, 'base_weights', GPT2_NAME), weights_only = True)
    assert torch.equal(dense.out_head.weight, params['out_head.weight']), 'out_head no está cargado'
    assert torch.equal(dense.final_norm.scale, params['final_norm.scale']), 'final_norm no está cargado'
    del params

    # Cada experto con su propia copia
    e = moe.moe_layers()[0].experts
    assert e[0].l1.weight.data_ptr() != e[1].l1.weight.data_ptr(), 'los expertos comparten memoria (falta .clone())'

    # num_heads distinto de 12 rechazado
    try:
        Utils.load_weights(GPTMOE(make_attr(heads = 8)), GPT2_NAME, base, blocks = 2)
    except ValueError:
        return
    raise AssertionError('load_weights acepta num_heads = 8 (falta el check del bloque 2, punto 3)')


# ---------------------------------------------------------------- bloque 3

@section('Bloque 3 · Congelado por grupos')
def _():
    model = small_model()

    def trainable(groups):
        Utils.freeze_selective(model, groups)
        return {n for n, p in model.named_parameters() if p.requires_grad}

    router = trainable(['router'])
    experts = trainable(['experts'])
    ffn = trainable(['ffn'])
    base = trainable(['base'])
    every = trainable(['base', 'ffn', 'experts', 'router'])

    assert router and all('.ff.gate.' in n for n in router), "'router' no deja solo el gate"
    assert experts and all('.ff.experts.' in n for n in experts), "'experts' no deja solo los expertos"
    assert ffn and all('.ff.ffn.' in n for n in ffn), "'ffn' no deja solo las capas FF"
    assert not any('.ff.' in n for n in base), "'base' incluye partes de ff"
    assert len(router) + len(experts) + len(ffn) + len(base) == len(every), 'los grupos se solapan'
    assert every == {n for n, _ in model.named_parameters()}, 'hay parámetros fuera de los 4 grupos'

    try:
        Utils.freeze_selective(model, ['routers'])
    except ValueError:
        return
    raise AssertionError('freeze_selective acepta un grupo inexistente')


# ---------------------------------------------------------------- bloque 4

@section('Bloque 4 · Configuración')
def _():
    E = config.EXPERIMENTS
    assert [len(E[k]['stages']) for k in ('E1', 'E2', 'E3')] == [7, 7, 1], 'número de etapas incorrecto'
    assert config.ATTR['moe_related']['experts'] == len(config.TASKS), 'experts != número de tareas'
    assert config.ATTR['attention_related']['num_heads'] == 12, 'num_heads debe ser 12 para GPT-2'
    assert config.ATTR['moe_mask']['schedule'] != 'all', "con 'all' no hay capas FF y E1 = E2"

    valid = {'base', 'ffn', 'experts', 'router'}
    for name, exp in E.items():
        assert exp['init'] in ('gpt2', 'random'), f"{name}: init '{exp['init']}'"
        for st in exp['stages']:
            assert set(st['groups']) <= valid, f"{name}/{st['name']}: grupos {st['groups']}"

    # Experto i entrenado con la tarea i
    for i, st in enumerate(E['E1']['stages'][:len(config.TASKS)]):
        assert st['expert'] == i and st['tasks'] == [config.TASKS[i]], f"etapa {st['name']} mal emparejada"
    json.dumps(E)


@section('Datos reales (data/)')
def _():
    missing = [t for t in config.TASKS if not os.path.exists(os.path.join('data', f'{t}.csv'))]
    if missing:
        raise Skip(f'faltan {missing}: generarlos con data/generator.ipynb')


# ---------------------------------------------------------------- bloque 5

@section('Bloque 5 · Loaders por etapa')
def _():
    utils_mod.DATA_DIR = DATA_TMP

    train, val, test = Utils.get_loaders(['task_a'], batch_size = 4, context_length = 64)
    batch = next(iter(train))
    assert batch['input_ids'].shape[0] == 4, 'tamaño de batch incorrecto'
    assert batch['input_ids'].shape == batch['target_ids'].shape, 'input y target con formas distintas'
    assert train.dataset.task_names == ['task_a'], 'el loader de una tarea carga otras'

    both = Utils.get_loaders(['task_a', 'task_b'], batch_size = 4, context_length = 64)[0]
    assert both.dataset.task_names == ['task_a', 'task_b'], 'no carga todas las tareas pedidas'


@section('Pérdidas (compañero)')
def _():
    from gpt2_moe.train_test.losses import LossSelector

    model = small_model(n_layers = 2)
    x = torch.randint(0, 50257, (2, 8))
    pending = []

    for name, params in (('ce', {}), ('own', {'moe_layers': model.moe_layers()})):
        try:
            loss_fn = LossSelector.loss_selector(name, params)
            if isinstance(loss_fn, type):
                pending.append(f"'{name}' devuelve la clase, no una instancia")
                continue
            loss = loss_fn(model(x).flatten(0, 1), x.flatten())
            if not (torch.is_tensor(loss) and loss.requires_grad):
                pending.append(f"'{name}' no devuelve un tensor con gradiente")
        except Exception as e:
            pending.append(f"'{name}': {type(e).__name__}: {e}")

    if pending:
        raise Pending('; '.join(pending) + ' -> el trainer no puede usarlas todavía')


# ---------------------------------------------------------------- bloque 6

class StubSelector:
    """ Sustituye a LossSelector en la prueba del runner: CE para todo """
    @staticmethod
    def loss_selector(name, params):
        return nn.CrossEntropyLoss()

@section('Bloque 6 · Trainer y runner (mini experimento)')
def _():
    utils_mod.DATA_DIR = DATA_TMP

    # 2 capas (0 MoE, 1 FF), 2 expertos, 3 etapas como en E1
    attr = make_attr(emb_dim = 64, heads = 4, context = 64, experts = 2)
    stages = [
        {'name': 'expert_0', 'groups': ['experts'], 'expert': 0, 'tasks': ['task_a'], 'loss': 'ce'},
        {'name': 'expert_1', 'groups': ['experts'], 'expert': 1, 'tasks': ['task_b'], 'loss': 'ce'},
        {'name': 'router_ffn', 'groups': ['router', 'ffn'], 'expert': None, 'tasks': None, 'loss': 'own'},
    ]
    train_mod.ATTR = attr
    train_mod.TRAIN = {'epochs': 1, 'bs': 4, 'lr': 1e-3}
    train_mod.TASKS = ['task_a', 'task_b']
    train_mod.EXPERIMENTS = {'MINI': {'init': 'random', 'stages': stages}}
    train_mod.BASE_PATH = os.path.join(TMP, 'checkpoints')
    train_mod.LossSelector = StubSelector

    model = train_mod.run('MINI', seed = 0, device = 'cpu')
    run_path = os.path.join(TMP, 'checkpoints', 'MINI', 'seed_0')

    # Un .pth y un historial por etapa
    for st in stages:
        for suffix in ('.pth', '_history.json'):
            assert os.path.exists(os.path.join(run_path, st['name'] + suffix)), f"falta {st['name']}{suffix}"

    # Cada etapa solo cambia lo que entrena
    torch.manual_seed(0)
    init = GPTMOE(attr).state_dict()
    load = lambda n: torch.load(os.path.join(run_path, f'{n}.pth'), weights_only = True)
    e0, e1, rf = load('expert_0'), load('expert_1'), load('router_ffn')

    def only(changed, prefixes, stage):
        assert changed, f'{stage}: no ha cambiado ningún parámetro'
        bad = sorted(k for k in changed if not any(p in k for p in prefixes))
        assert not bad, f'{stage}: cambian parámetros que deberían estar congelados: {bad[:3]}'

    only(changed_keys(init, e0), ['.ff.experts.0.'], 'expert_0')
    only(changed_keys(e0, e1), ['.ff.experts.1.'], 'expert_1')
    changed = changed_keys(e1, rf)
    only(changed, ['.ff.gate.', '.ff.ffn.'], 'router_ffn')
    assert any('.ff.gate.' in k for k in changed), 'router_ffn: el router no ha cambiado'
    assert any('.ff.ffn.' in k for k in changed), 'router_ffn: la FF no ha cambiado'

    # Al terminar no queda ningún experto forzado
    assert all(m.forced_expert is None for m in model.moe_layers()), 'queda un experto forzado tras run'

    # Relanzar -> se salta todo, no se reescribe nada
    mtimes = {f: os.path.getmtime(os.path.join(run_path, f)) for f in os.listdir(run_path)}
    train_mod.run('MINI', seed = 0, device = 'cpu')
    assert mtimes == {f: os.path.getmtime(os.path.join(run_path, f)) for f in os.listdir(run_path)}, \
        'al relanzar se vuelven a entrenar etapas ya hechas'

    if "__main__" not in open(train_mod.__file__).read():
        WARNINGS.append("train.py no tiene el bloque `if __name__ == '__main__'`: "
                        "`python -m gpt2_moe.train_test.train E1` no hará nada.")


# ---------------------------------------------------------------- resumen

shutil.rmtree(TMP, ignore_errors = True)

print('\n' + '=' * 70)
for title, status, msg in RESULTS:
    print(f'{status:<10} {title}')
    if msg:
        print(f'{"":<10} -> {msg}')
for w in WARNINGS:
    print(f'\nAVISO: {w}')
print('=' * 70)
