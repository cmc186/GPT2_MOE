# GPT-2 Mixture of Experts: estudio de la especialización inducida

Implementación de un modelo de lenguaje autorregresivo tipo **GPT-2** con capas **Mixture of Experts (MoE)** y evaluación experimental de cómo influye la estrategia de entrenamiento en la especialización de sus expertos.

Proyecto desarrollado para la **Práctica 1 de Procesamiento del Lenguaje Natural I (curso 2026–2027)** de la **Universidad de Alicante**, a partir de la arquitectura GPT presentada en *Build a Large Language Model (From Scratch)*, de Sebastian Raschka.

**Autores:** Carlos Molera Canals, Pablo García Beltrá, Linxi Jiang y Manuel García Romo.

## 1. Objetivo

La práctica propone comprender una implementación básica de un modelo de lenguaje y **ampliar su arquitectura con una técnica utilizada en LLM modernos**, justificando su funcionamiento mediante evidencias experimentales.

En este proyecto se ha elegido **Mixture of Experts**. En lugar de utilizar una única red *feed-forward* (FFN) en cada bloque Transformer, determinadas capas incorporan varios expertos FFN y un *router* que selecciona qué expertos procesan cada token.

El objetivo experimental no es únicamente comprobar que el modelo aprende las tareas, sino estudiar una cuestión adicional: **¿se especializan los expertos en función de la tarea y afecta a esa especialización entrenarlos por separado antes de entrenar el modelo completo?** Para responderla se compara un modelo denso con dos modelos MoE que comparten arquitectura, pero difieren en su procedimiento de entrenamiento.

[Enunciado de la práctica](https://jaspock.github.io/plni-2627/docs/practicallm.html) · [Código base de Raschka](https://github.com/rasbt/LLMs-from-scratch)

## 2. Arquitectura implementada

El proyecto implementa en **PyTorch** un Transformer *decoder-only* de tipo GPT-2, compuesto por embeddings de token y posición, atención multi-cabeza con máscara causal, normalización, conexiones residuales, bloques FFN y una proyección final al vocabulario. La principal ampliación permite sustituir las FFN convencionales por capas MoE de forma configurable.

### 2.1. Capa Mixture of Experts

Cada capa MoE contiene:

- **Expertos:** redes FFN independientes, con dos proyecciones lineales y activación GELU.
- **Router:** proyección lineal seguida de *softmax* que estima una distribución de probabilidad sobre los expertos para cada token.
- **Selección Top-k:** solamente los `k` expertos mejor puntuados procesan cada token; sus salidas se agregan ponderadas por los pesos del router.

Para `k > 1`, los pesos elegidos se renormalizan entre los expertos seleccionados. La selección se realiza **por token**, no por secuencia completa. La clase `MOE` permite además forzar un experto concreto, funcionalidad utilizada durante el entrenamiento individual.

La inserción de MoE es configurable mediante `moe_mask`: `none` (modelo denso), `all` (todas las capas), `alternate` (una capa de cada cierto intervalo) o una lista explícita de índices.

### 2.2. Configuración comparada

| Característica | E1: denso | E2 / E3: MoE |
|---|---:|---:|
| Capas Transformer | 10 | 10 |
| Dimensión de embedding | 512 | 512 |
| Cabezas de atención | 8 | 8 |
| Longitud de contexto | 64 | 64 |
| Vocabulario GPT-2 | 50.257 | 50.257 |
| Capas MoE | 0 | 5 (índices 0, 2, 4, 6 y 8) |
| Expertos por capa MoE | — | 6 |
| Expertos activos por token y capa MoE | — | 2 (*Top-2*) |
| Dimensión oculta FFN / experto MoE | 2.048 | 1.024 por experto |
| Parámetros totales | 83.020.800 | 104.040.990 |

En las capas MoE, la dimensión oculta de cada experto se ajusta en función de `k` (`4 × emb_dim / k`). Así, las dos FFN activas por token tienen una dimensión oculta total comparable a la FFN densa, aunque **el modelo MoE tiene más parámetros almacenados** y añade el coste del enrutamiento. No se presupone una mejora de velocidad o memoria: eso requeriría un *benchmark* específico.

Las tres configuraciones experimentales se inicializan **aleatoriamente**. El código también contiene utilidades para cargar pesos GPT-2 preentrenados y realizar *upcycling*, pero esas opciones no forman parte de los tres experimentos cuyos resultados se presentan aquí.

## 3. Datos y tareas

Se utiliza un conjunto **sintético y supervisado**, generado con `src/gpt2_moe/data/generator.ipynb`, con seis tareas de respuesta determinista:

| Tarea (`task`) | Qué debe aprender el modelo |
|---|---|
| `add_sub` | Suma y resta de enteros |
| `mul_div` | Multiplicación y división exacta |
| `alphabetic` | Ordenar alfabéticamente una secuencia de letras |
| `spell` | Separar los caracteres de una cadena mediante espacios |
| `parity` | Devolver `0` si un número es par y `1` si es impar |
| `first_last` | Devolver el primer y el último carácter de una cadena |

El generador contempla **10.000 ejemplos por tarea** en su ejecución incluida, con semilla `42`. Los archivos CSV contienen las columnas `prompt` e `y`. Se eliminan duplicados y ejemplos demasiado largos, por lo que el número final de muestras puede ser inferior al generado.

El modelo recibe secuencias con la forma:

```text
add_sub: 12 + 7 = 19<|endoftext|>
parity: 134 = 0<|endoftext|>
```

La tokenización utiliza el BPE de GPT-2 mediante `tiktoken`. Cada tarea se divide de forma reproducible en **80 % entrenamiento, 10 % validación y 10 % prueba**, antes de tokenizar. Se aplica *padding* por la derecha; la función de pérdida ignora tanto los tokens del enunciado como los de relleno (`ignore_index = -100`) y se calcula sobre la **respuesta y el token de fin de secuencia**.

## 4. Experimentos

Se comparan tres enfoques:

| Experimento | Modelo | Entrenamiento |
|---|---|---|
| **E1 — Dense baseline** | GPT denso, sin MoE | Entrenamiento conjunto de las seis tareas desde cero; 50 épocas, *batch size* 32. |
| **E2 — MoE con especialización inducida** | MoE alterno, 6 expertos por capa y Top-2 | Primero se entrena cada experto por separado con una tarea asignada (50 épocas por experto, *batch size* 32), manteniendo el resto congelado y forzando su selección. Después se entrenan conjuntamente modelo y router durante 15 épocas. |
| **E3 — MoE conjunto** | Misma arquitectura MoE que E2 | Entrenamiento conjunto desde el inicio, sin preentrenamiento individual de expertos; 15 épocas, *batch size* 64. |

Los tres utilizan `AdamW`, tasa inicial de aprendizaje `3e-4`, planificación `CosineAnnealingLR`, recorte de la norma del gradiente a `1.0` y la semilla `42` en la evaluación documentada.

**E2 y E3 comparten arquitectura, pero no presupuesto de entrenamiento:** E2 incluye seis etapas adicionales de entrenamiento supervisado. Por ello, la comparación permite estudiar diferencias observadas entre procedimientos, pero no aislar exclusivamente el efecto de la inicialización de los expertos.

### Función de pérdida

En el modelo denso se emplea entropía cruzada (*cross-entropy*). En el entrenamiento conjunto de los modelos MoE se combina con una regularización auxiliar de balanceo:

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{CE}}+\alpha\sum_{\ell\in\mathrm{MoE}}N\sum_{i=1}^{N}f_{\ell,i}\,p_{\ell,i}
$$

Aquí, `f` representa la fracción de asignaciones a cada experto, `p` la probabilidad media del router, `N` el número de expertos y `α = 0.01` en las configuraciones MoE. El balanceo se calcula sobre posiciones de respuesta no enmascaradas. Durante el entrenamiento individual de expertos se utiliza únicamente entropía cruzada.

El entrenamiento guarda el mejor *checkpoint* de cada etapa según el **exact match medio de validación**, además del historial de pérdidas, puntuaciones, tasa de aprendizaje y normas del gradiente.

## 5. Resultados

Los siguientes resultados proceden del **conjunto de test evaluado en `experiments.ipynb`**, con semilla `42`. Son resultados registrados en el notebook, **no una reproducción ejecutada desde este README**.

### 5.1. Calidad de respuesta: exact match

Porcentaje de respuestas generadas de forma *greedy* que coinciden exactamente con la respuesta esperada (ignorando espacios exteriores):

| Tarea | E1: denso | E2: MoE especializado | E3: MoE conjunto |
|---|---:|---:|---:|
| `add_sub` | **17,9 %** | 15,7 % | 17,2 % |
| `mul_div` | 16,5 % | **21,9 %** | 17,8 % |
| `alphabetic` | 100,0 % | 100,0 % | 100,0 % |
| `spell` | 98,8 % | 97,8 % | **99,0 %** |
| `parity` | 100,0 % | 100,0 % | 100,0 % |
| `first_last` | **99,4 %** | 99,3 % | **99,4 %** |
| **Media entre tareas** | **72,10 %** | **72,45 %** | **72,23 %** |

La **entropía cruzada media por token de respuesta** en test fue `0,4552` (E1), `0,4768` (E2) y `0,4468` (E3); en esta métrica, menor es mejor.

Los modelos resuelven muy bien ordenación, paridad, deletreo y extracción del primer/último carácter, mientras que las tareas aritméticas siguen siendo considerablemente más difíciles. **No se observa una mejora general sustancial de exact match por introducir MoE**: las medias son muy próximas.

### 5.2. Especialización y enrutamiento

La evaluación no se limita a la precisión. `experiments.ipynb` incluye mapas de calor de **frecuencia de asignación tarea–experto**, trazas de expertos seleccionados por token y capa, visualizaciones de atención y una prueba contrafactual en la que se cambia únicamente el nombre de la tarea en el *prompt*.

![Distribución del enrutamiento por tarea, experto y capa, para E2 y E3](src/images/router_hmaps/routing_heatmap.png)

En las primeras capas, **E2 muestra preferencias de enrutamiento más marcadas por tarea**. En E3 las asignaciones suelen aparecer más repartidas. Esta diferencia se atenúa o cambia entre capas: no implica que exista una asignación exclusiva y fija de cada tarea a un experto durante toda la red.

En el experimento contrafactual se compara el experto Top-1 asignado a los tokens compartidos de un mismo *prompt* al sustituir su identificador de tarea por el de otras tareas:

| Modelo | Proporción media de asignaciones Top-1 que cambian |
|---|---:|
| **E2: entrenamiento previo por experto** | **79,5 %** |
| **E3: entrenamiento conjunto** | **38,1 %** |

Estos resultados, calculados sobre los ejemplos concretos incluidos en el notebook, sugieren que **el router de E2 es más sensible al identificador explícito de tarea**. No constituyen por sí solos una demostración de especialización funcional independiente del nombre de la tarea.

También se implementa una métrica de especialización basada en la **divergencia de Jensen–Shannon** entre distribuciones de enrutamiento por tarea (`routing_jsd`), aunque el *benchmark* numérico guardado compara explícitamente `exact_match` y `token_ce`.

### 5.3. Conclusión experimental

La arquitectura MoE está integrada y permite enrutamiento Top-k, selección configurable de capas y entrenamiento por etapas. El principal hallazgo del estudio no es una mejora decisiva de calidad, sino que **la estrategia de entrenamiento modifica notablemente el patrón de selección de expertos**, aun cuando el rendimiento final de las tres configuraciones es parecido. Para atribuir causalidad con mayor rigor serían necesarias varias semillas, presupuestos de cómputo equiparables y experimentos adicionales sin identificadores explícitos de tarea.

## 6. Estructura del repositorio

```text
.
├── README.md
├── requirements.txt
├── documentation.tex                  # Borrador de la memoria académica
└── src/
    ├── gpt2_moe/
    │   ├── data/
    │   │   ├── generator.ipynb         # Generación de las seis tareas sintéticas
    │   │   ├── dataset.py             # Construcción, partición y enmascarado
    │   │   ├── dataloader.py          # Batches y padding
    │   │   └── tokenizer.py           # BPE de GPT-2
    │   ├── model/
    │   │   ├── gpt2_moe.py            # Modelo GPTMOE
    │   │   ├── tf_block.py            # Bloque Transformer FFN/MoE
    │   │   ├── att_block.py           # Atención causal multi-cabeza
    │   │   ├── ffn.py                 # Feed-forward densa
    │   │   ├── moe/                   # Expert, Router y MOE
    │   │   └── model_utils/           # Selección de capas MoE
    │   └── train_test/
    │       ├── train.py               # Entrenamiento y checkpoints
    │       ├── experiments.ipynb      # Benchmark y análisis del routing
    │       ├── configs/               # YAML de los experimentos
    │       ├── losses/                # CE y pérdida de balanceo
    │       ├── scorers/               # Exact match, CE, accuracy, JSD
    │       └── utils/                 # Carga de pesos, loaders, congelación
    ├── images/                         # Heatmaps de routing, atención y trazas
    └── results/                        # Resultados serializados de test
```

Otros notebooks del repositorio documentan el desarrollo y las comprobaciones de los componentes (`transformer.ipynb`, `moe_block.ipynb`, `balanced_loss.ipynb`, etc.).

## 7. Instalación y ejecución

Se necesita **Python 3.10 o superior** y se recomienda una GPU CUDA para los experimentos completos.

Desde la raíz del repositorio:

```bash
git clone https://github.com/cmc186/GPT2_MOE.git
cd GPT2_MOE

python -m venv .venv
source .venv/bin/activate       # En Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Dependencias adicionales utilizadas por la configuración y los notebooks:
pip install PyYAML numpy matplotlib seaborn notebook
```

### 7.1. Generación de datos

El repositorio **no incluye los CSV de entrenamiento** (la carpeta `data/` está excluida por `.gitignore`). Hay que generarlos antes de entrenar:

1. Abrir `src/gpt2_moe/data/generator.ipynb` en Jupyter o VS Code.
2. Ejecutar sus celdas con el directorio de trabajo del notebook situado en `src/gpt2_moe/data/`.
3. Comprobar que se han creado los seis CSV en **`data/` de la raíz del repositorio**. La variable `DATA_PATH` utiliza una ruta relativa (`../../../data`) y debe ajustarse si cambia el directorio de trabajo.

**Importante:** si ya existen los CSV, el generador **añade filas** en lugar de reemplazarlos. Para comenzar una generación limpia, utiliza una carpeta vacía.

### 7.2. Entrenar los modelos

Desde la carpeta `src/`, que permite resolver el paquete `gpt2_moe`:

```bash
cd src

# E1: referencia densa
python -m gpt2_moe.train_test.train --name test_1.yaml --dev cuda --seed 42

# E2: preentrenamiento por experto seguido de entrenamiento conjunto
python -m gpt2_moe.train_test.train --name test_2.yaml --dev cuda --seed 42

# E3: MoE entrenado conjuntamente desde cero
python -m gpt2_moe.train_test.train --name test_3.yaml --dev cuda --seed 42
```

Sustituye `cuda` por `cpu` si no tienes GPU, teniendo en cuenta que el entrenamiento será más lento. Las configuraciones están en `src/gpt2_moe/train_test/configs/cf_files/`.

Los pesos e historiales se almacenan en:

```text
src/gpt2_moe/checkpoints/<nombre_experimento>/seed_42/
    expert_0.pth ... expert_5.pth    # Solo en E2
    full.pth
    full_history.json
```

El programa reutiliza checkpoints existentes y omite las etapas que ya dispongan de un archivo `.pth`. Para comenzar un entrenamiento desde cero, se debe usar un directorio de experimento nuevo o retirar manualmente los checkpoints anteriores.

### 7.3. Evaluación y visualización

Abrir `src/gpt2_moe/train_test/experiments.ipynb` y ejecutar las celdas después de haber generado los datos y obtenido los checkpoints. El notebook permite comparar pérdidas, `exact_match`, entropía cruzada por token, rutas de expertos, patrones de atención y sensibilidad del router a la etiqueta de tarea.

**Nota de reproducibilidad:** el notebook de evaluación carga E2 desde un directorio llamado `E2_scratch_moe_alternate`, mientras que `test_2.yaml` define actualmente `E2_scratch_moe_alternate_local`. Para evaluar checkpoints creados con esa configuración hay que adaptar el nombre del directorio en el notebook. Los checkpoints históricos utilizados para producir las salidas guardadas en el notebook **no forman parte del repositorio**.

## 8. Referencias

- Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. [Código](https://github.com/rasbt/LLMs-from-scratch).
- Shazeer et al. (2017). [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538).
- Fedus, Zoph y Shazeer (2022). [*Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*](https://arxiv.org/abs/2101.03961).
- Komatsuzaki et al. (2023). [*Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints*](https://arxiv.org/abs/2212.05055).
- Jiang et al. (2024). [*Mixtral of Experts*](https://arxiv.org/abs/2401.04088).
