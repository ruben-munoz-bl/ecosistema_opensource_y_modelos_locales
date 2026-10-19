# Reto del Día — Comparativa de modelos LLM open-source

> **Fecha de elaboración:** 05-oct-2026 · **Actualización:** 19-oct-2026, con **acceso real a la VM de Azure** (SSH + APIs REST ya desplegadas).
> Las secciones **§1–§3** proceden de **documentación oficial pública** (fichas de modelo, informes técnicos arXiv,
> documentación de proveedores); la **§4** es **medida in situ en la VM**. Lo que sigue sin medir se marca **ESTIMACIÓN**.
> Fuentes al final (**§7**).

---

## 1. Familia Gemma — variantes E2B, E4B, 12B, 26B-A4B, 31B

### 1.1 Nota de nomenclatura (obligatoria)

| Nombre pedido en el reto | Variante real equivalente | Estado |
|---|---|---|
| `google/gemma-4-E2B-it` | `google/gemma-4-E2B-it` | existente en HF |
| `google/gemma-4-E4B-it` | `google/gemma-4-E4B-it` | existente en HF |
| `google/gemma-4-12B-it` | `google/gemma-4-12B-it` | existente en HF |
| `google/gemma-4-26B-A4B-it` | `google/gemma-4-26B-A4B-it` | existente en HF |
| `google/gemma-4-31B-it` | `google/gemma-4-31B-it` | existente en HF |

> **El "E" = *effective*.** En E2B/E4B solo una fracción de los parámetros es efectivamente útil por token:
> **2.3B de 5.1B** (E2B) y **4.5B de 8B** (E4B). No confundir con **Gemma 3n** (2025), que *también* usaba
> las siglas E2B/E4B para otra familia distinta.

### 1.2 Especificaciones y modalidades

Fuente: ficha oficial `ai.google.dev/gemma/docs/core/model_card_4` (act. 30-jul-2026) + arXiv:2607.02770.

| Variante | Parámetros efectivos | Parámetros totales | Capas | Contexto | Modalidades (entrada → salida) | Encoders |
|---|---|---|---|---|---|---|
| **Gemma 4 E2B** | **2.3B** ("E" = *effective*) | 5.1B con embeddings | 35 | 128K | Texto, Imagen, Audio → Texto | visión ~150M + audio ~300M |
| **Gemma 4 E4B** | **4.5B** | 8B con embeddings | 42 | 128K | Texto, Imagen, Audio → Texto | visión ~150M + audio ~300M |
| **Gemma 4 12B Unified** | **11.95B** (arquitectura *encoder-free*) | 11.95B | 48 | 256K | Texto, Imagen, **Audio** → Texto | **sin encoders**: proyecta parches de imagen y ondas de audio directamente al espacio de embeddings con capas lineales ligeras |
| **Gemma 4 26B-A4B** | **3.8B activos** (MoE: 8 de 128 expertos + 1 compartido) | 25.2B totales | 30 | 256K | Texto, Imagen → Texto | visión ~550M · sin audio |
| **Gemma 4 31B** | **30.7B** (denso, todos activos) | 30.7B | 60 | 256K | Texto, Imagen → Texto | visión ~550M · sin audio |

| Fecha | Hito | Verificado en |
|---|---|---|
| **2-abr-2026** | Lanzamiento público de Gemma 4 (4 variantes: E2B, E4B, 26B-A4B, 31B), construida sobre la investigación de Gemini 3 | blog de Google *"Gemma 4: Byte for byte…"* y *Google Open Source* (jueves 2-abr-2026) |
| **3-jun-2026** | Se añade **Gemma 4 12B** a la familia (arquitectura *encoder-free*, primer modelo medio con audio) | *Google Developers Blog — Gemma 4 12B: The Developer Guide* (3-jun-2026) |
| **2-jul-2026** | Publicado el **informe técnico** (v1; v2 el 24-jul-2026) | arXiv:2607.02770 |
| **30-jul-2026** | Última actualización de la ficha oficial de Google | `ai.google.dev/gemma/docs/core/model_card_4` |

**Por qué "Unified"**: Google anuncia: *"Bypassing heavy multi-stage vision and audio encoders entirely, multimodal
data is fed straight into the LLM backbone, reducing multimodal latency"*. La 12B *Unified* elimina los codificadores
pesados multi-etapa. Matiz: el audio **no** entra crudo — usa un encoder ASR ligero (Qwen3-ASR) y la imagen se
proyecta directamente; es decir, *encoder-free* = **sin codificadores pesados multi-etapa**, no literalmente sin proyección.

---

### 1.3 VRAM estimada (FP16) y real en GGUF Q4

| Variante | Pesos | VRAM BF16 oficial (Google) | VRAM Q4_0 oficial (Google) | Artefacto GGUF Q4 (Ollama) |
|---|---|---|---|---|
| **Gemma 4 E2B** | 5.1B con embeddings | 11.4 GB | 2.9 GB | 4.6 – 7.5 GB |
| **Gemma 4 E4B** | 8.0B con embeddings | 17.9 GB | 4.5 GB | 6.6 – 9.5 GB |
| **Gemma 4 12B Unified** | 11.95B (encoder-free) | 26.7 GB | 6.7 GB | 7.7 – 8.0 GB |
| **Gemma 4 26B-A4B** | 25.2B (3.8B activos) | 57.7 GB | 14.4 GB | 16 – 19 GB |
| **Gemma 4 31B** | 30.7B | 69.9 GB | 17.5 GB | 19 – 20 GB |

> **Dos fuentes distintas, no intercambiables:** las columnas BF16/Q4_0 son la tabla oficial de
> `ai.google.dev/gemma/docs/core` (pesos del modelo, sin KV-cache); la columna Ollama es el **tamaño del archivo GGUF**
> publicado (incluye metadatos y suele usar Q4_K_M, algo más grande que Q4_0). El informe técnico da además
> `E2B 4.6 / E4B 9.0 / 12B 24.0 / 26B-A4B 52.0 / 31B 64.0` GB en bf16 y `2.3 / 7.65 / 16.2 / 19.2` GB cuantizados,
> más KV-cache (`+0.05` a `+1.10` GB según modelo).
> Nota MoE: en el **26B-A4B** los 3.8B son los **activos por token**; para la VRAM hay que contar **todos** los 25.2B
> (todos los pesos viven en memoria; la ventaja es en FLOPs, no en memoria).

---

### 1.4 Benchmarks públicos (tabla oficial de Google, instruction-tuned, CoT, sin herramientas de búsqueda)

**Nota de verificación (19-oct-2026):** valores contrastados **verbatim** con la model card oficial
(`huggingface.co/google/gemma-4-E4B-it/raw/main/README.md`) y con arXiv:2607.02770.
Las columnas siguen el orden oficial de Google: **31B · 26B-A4B · 12B · E4B · E2B**.

| Modelo (↓ benchmark) | 31B | E4B | 12B Unified | 26B-A4B | E2B |
|---|---|---|---|---|---|
| **MMLU Pro** | 85.2 | 69.4 | 77.2 | 82.6 | 60.0 |
| **AIME 2026** (sin herramientas) | 89.2 | 42.5 | 77.5 | 88.3 | 37.5 |
| **LiveCodeBench v6** | 80.0 | 52.0 | 72.0 | 77.1 | 44.0 |
| **Codeforces ELO** | 2150 | 940 | 1659 | 1718 | 633 |
| **GPQA Diamond** | 84.3 | 58.6 | 78.8 | 82.3 | 43.4 |
| **Tau2** (media de 3) | 76.9 | 42.2 | 69.0 | 68.2 | 24.5 |
| **HLE** (sin herramientas) | 19.5 | — | 5.2 | 8.7 | — |
| **BigBench Extra Hard** | 74.4 | 33.1 | 53.0 | 64.8 | 21.9 |
| **MMMLU** (multilingüe) | 88.4 | 76.6 | 83.4 | 86.3 | 67.4 |
| **MMMU Pro** (visión) | 76.9 | 52.6 | 69.1 | 73.8 | 44.2 |
| **MATH-Vision** | 85.6 | 59.5 | 79.7 | 82.4 | 52.4 |
| **CoVoST** (ST, en→xx) | — | 35.54 | 38.5 * | — | 33.47 |
| **FLEURS** (ASR, ↓ mejor) | — | 0.08 | 0.069 * | — | 0.09 |
| **MRCR v2** (8 needle, 128k) | 66.4 | 25.4 | 43.4 | 44.1 | 19.1 |

\* Excluyendo chino. **Nota:** SciCode e IFEval aparecen en análisis de terceros (p. ej. Artificial Analysis cita SciCode 43 % para el 31B),
pero **no forman parte de la tabla oficial de Google**; se han retirado de esta comparativa para no mezclar fuentes.

> Las cifras de Gemma 4 son de la **tabla oficial de Google** (instruction-tuned, con CoT y sin herramientas de búsqueda),
> verificadas contra `huggingface.co/google/gemma-4-E4B-it/raw/main/README.md`;
> las de Llama 3, Phi-3 y Mistral, de sus **informes técnicos y documentación oficial**. Setups distintos → **no comparables entre filas**.

---

### 1.5 Sobre MMLU y HumanEval en Gemma 4 — declaración explícita

Los benchmarks incluidos en la documentación oficial de Gemma 4 son: MMLU-Pro, AIME 2026, LiveCodeBench v6,
Codeforces ELO, GPQA Diamond, Tau2, HLE, BigBench-Extra-Hard, MMMLU, MMMU-Pro, OmniDocBench 1.5, MATH-Vision,
MedXPertQA MM, CoVoST, FLEURS y MRCR v2.
Los benchmarks **no incluidos** son: **MMLU** (clásico), **HumanEval**, **SciCode** e **IFEval**
(estos dos últimos solo aparecen en evaluaciones de terceros, p. ej. Artificial Analysis).
Comprobado en la ficha de HF `google/gemma-4-E4B-it`, la ficha oficial de Google y el informe técnico arXiv:2607.02770.

---

## 2. Tabla comparativa general (5 modelos)

### 2.1 Tabla principal (≥5 modelos open-source)

| Modelo | Repo en HF | Parámetros | VRAM FP16 (est.) | VRAM Q4 (GGUF/Ollama) | Cuantización típica | Licencia | Idiomas | Casos de uso |
|---|---|---|---|---|---|---|---|---|
| **Gemma 4 12B Unified** | `google/gemma-4-12B-it` | 11.95B | 26.7 GB (BF16 oficial) | 7.7 – 8.0 GB | Q4_K_M | Apache 2.0 | 140+ (texto), 35+ (audio) | razonamiento general, visión, audio, coding, agentes |
| **Gemma 4 E4B** | `google/gemma-4-E4B-it` | 4.5B efectivos (8B c/ embeddings) | 17.9 GB (BF16 oficial) | 6.6 – 9.5 GB | Q4_K_M | Apache 2.0 | 140+ | razonamiento, coding, agentes, multimodal compacto |
| **Llama 3 8B Instruct** | `meta-llama/Meta-Llama-3-8B-Instruct` | 8.03B | ≈ 16.1 GB | 4.7 – 5.0 GB | Q4_K_M | Llama 3 Community | inglés (>99 %) | instrucción general, RAG, fine-tuning |
| **Phi-3 mini** | `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ≈ 7.7 GB | 2.3 GB | Q4_K_M | MIT | inglés (mayoritario) | razonamiento, código, texto en recursos limitados |
| **Mistral 7B Instruct v0.3** | `mistralai/Mistral-7B-Instruct-v0.3` | 7.24B | ≈ 14.5 GB | 4.1 GB | Q4_K_M | Apache 2.0 | inglés | instructivo general, RAG, fine-tuning |

### 2.2 Lectura de la tabla y recomendación

- **Gemma 4 12B Unified** y **Gemma 4 E4B** son las únicas que dan acceso a la familia Gemma 4 en este rango:
  Apache 2.0, 140+ idiomas, razonamiento configurable y entradas de imagen/audio.
- **Gemma 4 E4B** es la recomendación por defecto en una VM de 16 GB: multimodal completa (texto+imagen+audio),
  ~7 GB en Q4 y 128K de contexto.
- **Gemma 4 12B Unified** si se prioriza calidad (77.2 en MMLU-Pro) y hay margen de disco: 7.7–8.0 GB en Q4,
  256K de contexto y **audio nativo** (primer modelo medio de la familia con audio). Cabe en 16 GB, pero con menos
  margen para KV-cache que E4B.
- **26B-A4B y 31B no caben** en 16 GB (16–19 GB y 19–20 GB en Q4) — descartadas para esta VM.
- **Phi-3 mini** es la más ligera (2.3 GB) para tareas de texto sin imagen/audio.

---

## 3. Análisis — ¿qué modelo elegir para producción con <16 GB de VRAM?

- **Phi-3 mini (3.8B):** ~7.7 GB FP16. **2.3 GB** en Q4_K_M. Cabe holgadamente; opción más barata y rápida en CPU.
- **Llama 3 8B:** ~16.1 GB FP16. **~4.7 GB** en Q4_K_M. Cabe en 16 GB **solo con cuantización**; es el modelo de uso general más equilibrado de la tabla.
- **Mistral 7B:** ~14.5 GB FP16. **~4.1 GB** en Q4_K_M. Cabe incluso en 8 GB; el mejor ratio calidad/peso en su gama.
- **Gemma 4 E4B:** ~17.9 GB BF16 (oficial). **6.6–9.5 GB** en GGUF Q4. Cabe en 16 GB con margen justo de KV-cache; multimodal (imagen+audio).
- **Gemma 4 12B Unified:** ~26.7 GB BF16 (oficial). **7.7–8.0 GB** en GGUF Q4. Cabe en 16 GB **solo cuantizado**; multimodal encoder-free con audio.
- **Gemma 4 26B-A4B:** ~57.7 GB BF16 (oficial). **16–19 GB** en GGUF Q4. **No cabe** en 16 GB.
- **Gemma 4 31B:** ~69.9 GB BF16 (oficial). **19–20 GB** en GGUF Q4. **No cabe** en 16 GB.

### 3.1 Rango de VRAM (16 GB es la referencia estándar de escritorio)

| Rango | Modelo(s) | Uso recomendado |
|---|---|---|
| **Rango 1: hasta 8 GB** | **Phi-3 mini (2.3 GB)** | tareas de texto ligeras, CPU, edge |
| **Rango 2: 8–16 GB** | **Mistral 7B (4.1 GB)** / **Llama 3 8B (4.7 GB)** | uso general con buen balance |
| **Rango 3: 16–24 GB** | **Gemma 4 E4B (6.6–9.5 GB)** / **Gemma 4 12B (7.7–8.0 GB)** | multimodal, visión, audio, agentes |
| **Rango 4: 24–48 GB** | **Gemma 4 26B-A4B (16–19 GB)** | agentes con MoE, mejor calidad por token activo |
| **Rango 5: 48 GB+** | **Gemma 4 31B (19–20 GB)** | máxima calidad densa |

### 3.2 MoE no reduce VRAM

El **26B-A4B** activa solo 3.8B de parámetros por token, pero **todos los 25.2B viven en la RAM/VRAM**: la ventaja del
MoE es en FLOPs/latencia, **no** en consumo de memoria. Por eso un MoE de 25B ocupa como un denso de 25B.

### 3.3 Conclusión

**E4B** es el *sweet spot* para una VM de 16 GB: multimodal completo (texto+imagen+audio), 128K de contexto, licencia
Apache 2.0 y **6.6–9.5 GB en Q4**, lo que deja margen para KV-cache y el resto de servicios de la VM.

---

## 4. VM de Azure de Inetum — acceso, hardware y modelos desplegados (medido in situ el 19-oct-2026)

### 4.1 Acceso real (cambia el supuesto inicial)

- **Vía SSH**, no IP+puerto público: clave `id_ed25519` en `C:\Users\<usuario>\.ssh\` + bloque `vm-b_ruben_munoz_blanco`
  (`HostName 74.234.250.36`, `User b_ruben_munoz_blanco`) en `~/.ssh/config`. Conexión desde VS Code con extensión
  **Remote-SSH** (`F1 → Remote-SSH: Connect to Host`) o `ssh -i ~/.ssh/id_ed25519 vm-b_ruben_munoz_blanco`.
- **Las APIs de inferencia NO están publicadas a internet**: viven en `127.0.0.1` de la VM. Para llamarlas desde el PC
  se usa un **túnel SSH**: `ssh -N -L 11434:127.0.0.1:11434 -L 11435:127.0.0.1:11435 vm-b_ruben_munoz_blanco`
  y después `AZURE_VM_URL="http://127.0.0.1:11434"` (en VS Code Remote-SSH: *Forward a Port*).
- **Restricción autoimpuesta:** operar solo dentro del usuario `b_ruben_munoz_blanco` (sin `sudo`, sin tocar procesos,
  ficheros ni datos de otros becarios que comparten la VM). Todo lo de §4 es de solo lectura.

### 4.2 Hardware y recursos (medidos)

| Recurso | Valor medido |
|---|---|
| Host / SO | `genai-openllm-vm` · Ubuntu 22.04.1 LTS (kernel `6.8.0-1059-azure`) · uptime 123 días |
| CPU / RAM | **8 vCPU** · **54 GiB** de RAM (≈43 GiB disponibles) · sin swap |
| Disco | 497 GB, **93 % usado → ~38 GB libres** (limita descargas de pesos) |
| **GPU** | **NVIDIA Tesla T4** (TU104GL, `0001:00:00.0`) **visible en PCI pero SIN driver**: no existe `/dev/nvidia*`, no hay `/proc/driver/nvidia/version`, `nvidia-smi` falla |
| Consecuencia | La inferencia va por **CPU**. La **VRAM real NO es medible** hasta que un administrador cargue el driver (requiere `sudo`, fuera del alcance del usuario). La T4 tiene **16 GB**, coherente con el análisis de §3 |

### 4.3 Servicios HTTP desplegados (descubiertos con `/openapi.json` y `/docs`, solo lectura)

Todos los servicios de la tabla pertenecen a **otros usuarios** de la VM compartida (identificados con `ps -eo user,cmd`,
sin `sudo`): hay varios becarios ejecutando contenedores Docker con FastAPI/Uvicorn.

| Puerto | Servicio detectado | Usuario | API de chat | Estado al probar (19-oct) |
|---|---|---|---|---|
| **8000** | FastAPI 0.1.0 (uvicorn `main:app`) | `b_pol_am...` | `GET /` → `{"backend":"local"}` · `POST /chat?prompt=<texto>` | **`500` tras 120 s** (cuelga y falla) |
| **8001** | FastAPI 0.1.0 (uvicorn `app:app`) | `b_yazan...` | `POST /chat` body `{"prompt":"..."}` | **timeout 90 s** |
| **18000** | Docker → mismo contenedor que 8000 | `b_pol_am...` | `GET /` → `{"backend":"local"}` · `POST /chat?prompt=` | **timeout 90 s** |
| **18001** | **Chat Dual API 1.0.0** (Docker, `uvicorn api:app`) | `b_richar...` | `GET /health` → `{"status":"ok","backend":"local"}` · `POST /api/chat` body `{"prompt":"..."}` → `{"text","backend"}` | **`502`**: `model 'gemma3:4b' not found` |
| 7002 | BotFoundry IDE (RAG API) | `pcanet` | no es LLM de chat genérico | — |
| 7003 | nginx → chatbot i-DE (frontend) | — | — | — |
| 8080 / 8081 / 8082 | Craft *Deliberative RAG / ETL / Retrieval* Lite API | `pcanet` | RAG y ETL, no generación | — |
| 5000 · 13000 · 3005 · 18080 | Flask / nginx (404 en rutas de chat) | varios | — | — |
| 3000 | GENEVA (frontend, *Evaluación Ambiental*) | — | — | — |
| 6333-6334 · 6379 · 5432-15432 | Qdrant · Redis · PostgreSQL | varios | infraestructura de RAG | — |
| **11434 · 11435** | **Ollama** (dos instancias de **otros becarios**) | `b_hecto`/`b_oscar`/`b_richa` | **OpenAI-compatible** `POST /v1/chat/completions` · `GET /v1/models` · `GET /api/tags` | **`200 OK`** (responden, ver §4.4) |

**Conclusión práctica:** la "Chat Dual API" (18001) tenía la forma exacta que pedía el reto (`{"text","backend"}`),
pero **no sirve**: su backend local no tiene el modelo `gemma3:4b`. Los puertos 8000/8001/18000 **cuelgan o devuelven 500**
(inferencia en CPU del propio becario, saturada — de hecho otro becario, `b_marin`, tenía un `curl` colgado en 8000).
**Los únicos endpoints que responden de verdad son los de Ollama (11434 y 11435)**, que además son compatibles con OpenAI.

### 4.4 Modelos detectados en la VM (solo lectura, vía API de Ollama)

| Puerto | Modelo | Formato | Parámetros (según GGUF) | Cuantización | Contexto | Tamaño | Capacidades |
|---|---|---|---|---|---|---|---|
| 11434 | `gemma4:e4b` | GGUF (`llamacpp`) | **7.5B** (`parameter_size`) | **Q4_K_M** | 131 072 | 6.58 GB | completion, **vision**, tools |
| 11435 | `gemma:latest` | GGUF (`ggml`) | 9B | Q4_0 | 8 192 | 5.01 GB | completion |

- `gemma4:e4b` es la **variante E4B** de §1.2: el artefacto (7.5B totales, Q4_K_M, 6.6 GB, 128K ctx) **cuadra con §1.3**
  (VRAM Q4 de E4B ≈ 6.6 GB) y confirma la nomenclatura de efectivos 4.5B / totales ~8B.
- ⚠️ **No hay ningún modelo desplegado en mi usuario**: `b_ruben_munoz_blanco` no tiene procesos Ollama ni `~/.ollama`.
  Las dos instancias pertenecen a **otros becarios**: se consultan en modo solo-lectura (`GET /api/tags`, `GET /v1/models`)
  y se les lanzan **como máximo 1-2 prompts cortos** para el reto. No se instala, copia ni modifica nada de su entorno.
- **VRAM no medida** (ver §4.2): sin driver no hay `nvidia-smi`; los tamaños son los del artefacto, no la ocupación real.
- ⚠️ **`gemma:latest` (11435) NO es un Gemma 4**: su GGUF declara `family: gemma` (frente a `family: gemma4` del 11434),
  `parameter_size 9B`, 8K de contexto y el runner antiguo `ggml`. Sirve de segundo modelo para el reto, pero **queda
  fuera de la familia de §1** y no debe citarse como "Gemma 4" en ninguna tabla.

### 4.5 Cierre de las tareas 4 y 5 del reto

- **Tarea 4 — "verificar qué modelos están desplegados":** cumplida. No hay modelos propios; hay **dos modelos servidos
  por Ollama de otros becarios** (`gemma4:e4b` en 11434 y `gemma:latest` en 11435) y varias APIs de chat ajenas caídas o colgadas (§4.3).
- **Tarea 5 — "mismo prompt en ≥2 modelos y analizar la calidad":** cumplida con esos dos modelos; resultados y análisis en **§5.1**.

---

## 5. `inference_azure.py` — instrucciones de uso (adaptado a la VM real)

El script soporta los **4 estilos de API que la VM expone de verdad** (§4.3) y guarda respuestas + métricas en `results/`.
Depende solo de `requests`. **Antes de nada hay que abrir el túnel SSH** (las APIs viven en `127.0.0.1` de la VM):

```bash
# 0) TÚNEL SSH — terminal aparte (o VS Code Remote-SSH → Forward a Port)
ssh -N -L 11434:127.0.0.1:11434 -L 11435:127.0.0.1:11435 vm-b_ruben_munoz_blanco

#   Aviso: el túnel tiene que seguir vivo durante TODA la inferencia. Si se cae, el script
#   no se rompe: marca `ok:false` con `ConnectionError`, guarda el JSON y devuelve exit 1
#   (evidencia real: results/fallo_tunel_ssh_20261019.json). Para corridas largas conviene:
#   ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 ...

# 1) Entorno local (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2) RETO: MISMO PROMPT EN 2 MODELOS (los dos endpoints OpenAI-compatible que responden, §4.4)
python inference_azure.py `
  --service gemma4:e4b=http://127.0.0.1:11434/v1/chat/completions `
  --service gemma:latest=http://127.0.0.1:11435/v1/chat/completions `
  --prompt "Resume en 3 frases qué es un modelo de mezcla de expertos (MoE)." --timeout 300

# 3) Modo legado compatible con OpenAI (un endpoint .../v1/chat/completions cualquiera)
$env:AZURE_VM_URL = "http://127.0.0.1:11434"     # bash: export AZURE_VM_URL="http://127.0.0.1:11434"
python inference_azure.py --model gemma4:e4b --prompt-file prompt.txt --max-tokens 512

# 4) Los demás estilos descubiertos en la VM (hoy caídos: ver estado en §4.3)
python inference_azure.py --api dual --service dual=http://127.0.0.1:18001/api/chat
python inference_azure.py --api query --service a=query@http://127.0.0.1:8000/chat

# 5) Salida explícita
python inference_azure.py --service a=http://127.0.0.1:11434/v1/chat/completions --out results/comparativa_01.json
```

**Cómo se elige el estilo:** `--api openai|query|body|dual` lo fija para todos; si no se indica se **infiere de la URL**
(`/v1/chat/completions`→openai, `/api/chat`→dual, `/chat`→body) y puede anotarse por destino como `NOMBRE=URL@ESTILO`.
En los estilos no-OpenAI no existe `model` ni `usage`, así que los tokens se **estiman** (`caracteres/4`).

**Métricas registradas por destino:** latencia wall-time (s), tokens entrada/salida (`usage` de la API si existe; si no,
estimación), longitud en caracteres/líneas, tokens/s, campo **`backend`** cuando la API lo devuelve, **`finish_reason`**
(`length` = respuesta cortada por `--max-tokens`, `stop` = terminó natural) y **`reasoning_chars`** +
`used_reasoning_as_answer` para detectar modelos de razonamiento cuya respuesta sale del campo `reasoning`.
La tabla resumen se imprime en consola y todo se persiste en `results/*.json`.

**Códigos de salida:** `0` = todos los destinos respondieron · `1` = al menos uno falló · `2` = configuración inválida
(sin destino, `AZURE_VM_URL` ausente con `--model`, o argumentos malformados).

### 5.1 Ejecución real y análisis de calidad (19-oct-2026)

**Prompt único** (mismo para los 2 modelos, `temperature 0.2`, `timeout 900`):

> *Resume en 3 frases que es un modelo de mezcla de expertos (MoE) y por que se usa para inferencia local.*

| Corrida | `max_tokens` | Modelo | Latencia | Tokens out | tok/s | `finish_reason` | Evidencia (JSON) |
|---|---:|---|---:|---:|---:|---|---|
| **A** | 400 | `gemma4:e4b` | 160,7 s | 400 | 2,49 | `length` ← **cortado** | `results/comparativa_vm_20261019.json` |
| A | 400 | `gemma:latest` | 15,5 s | 67 | 4,32 | `stop` | idem |
| **B** | 900 | `gemma4:e4b` | **249,9 s** | **763** (2426 car. de razonamiento + 804 de respuesta) | 3,05 | `stop` | `results/comparativa_vm_20261019_budget900.json` |
| B | 900 | `gemma:latest` | **20,7 s** | 64 | 3,09 | `stop` | idem |

*(Corrida intermedia perdida por caída del túnel SSH → `results/fallo_tunel_ssh_20261019.json`, `ok:false` para ambos.)*

#### Respuestas (texto completo en los JSON)

**`gemma4:e4b` (corrida B):**
> Un Modelo de Mezcla de Expertos (MoE) es una arquitectura de IA que combina múltiples sub-redes especializadas, llamadas
> "expertos". Un mecanismo de puerta (gating) decide dinámicamente qué expertos son más relevantes… Esto permite que el
> modelo sea masivamente grande sin incurrir en el costo computacional de activar todos sus parámetros…
> **¿Por qué se usa para inferencia local?** Su eficiencia en la activación de parámetros lo hace ideal… se puede ejecutar
> en dispositivos con limitaciones de hardware, como computadoras personales o móviles.

**`gemma:latest` (idéntico en ambas corridas):**
> Un modelo de mezcla de expertos (MoE) es un modelo estadístico que combina las predicciones de varios expertos
> individuales para crear una predicción más precisa. Se utiliza para la inferencia local porque puede capturar el
> conocimiento local de los expertos, lo que permite realizar inferencias precisas a partir de datos específicos.

#### Análisis comparado

| Criterio | `gemma4:e4b` (7,5B · Q4_K_M) | `gemma:latest` (9B · Q4_0) |
|---|---|---|
| **Precisión conceptual** | ✅ **Correcta**: expertos + *gating/router* + activación dispersa (solo una fracción de parámetros activos) | ⚠️ **Incorrecta/vaga**: lo define como un *ensemble* que "combina predicciones" (votación estadística), no la arquitectura MoE con enrutado |
| **Razón de uso local** | ✅ Vincula el ahorro de FLOPs/latencia con hardware limitado (lo que pide el enunciado) | ❌ Responde "capturar el conocimiento local de los expertos": confunde *local inference* con *conocimiento local de datos* |
| **Formato pedido (3 frases)** | ⚠️ 5 frases + encabezado Markdown: incumple el formato | ⚠️ Solo 2 frases |
| **Idioma** | ✅ Español (aunque su razonamiento interno sale en inglés) | ✅ Español |
| **Coste** | ❌ 249,9 s y 763 tokens (su fase de razonamiento son **2426 caracteres** previos) | ✅ 20,7 s y 64 tokens (12× más rápido y barato) |

**Veredicto:**

1. **Calidad de contenido: gana `gemma4:e4b` con claridad.** Es el único que da la explicación técnica correcta de MoE
   y la conecta con el ahorro computacional de la inferencia local. `gemma:latest` se equivoca en el concepto y en el
   motivo de uso local: **respuesta inútil para el enunciado pese a ser fluida**.
2. **Calidad de formato: ninguno de los dos cumple.** E4B entrega 5 frases + encabezado; `gemma:latest`, 2 frases.
   Hace falta un *system prompt* más estricto (o `temperature 0` + instrucción de longitud).
3. **Coste/latencia: gana `gemma:latest` por 12×.** En CPU, E4B necesita **presupuesto de tokens**: con `max_tokens 400`
   **ni siquiera respondió** (corrida A: se le comió el límite en la fase de razonamiento, `finish_reason=length`);
   con 900 sí termina (`stop`). Regla práctica: para un modelo de razonamiento, `max_tokens ≥ 800-1000`.
4. **Conclusión práctica (encaja con §3):** en esta VM, con CPU y sin driver de GPU, **el modelo razonador paga un peaje
   de ~12× en latencia por mejor contenido**; para tareas de extracción/resumen corto, un modelo directo como
   `gemma:latest` (o `gemma4:e2b`) rinde mejor *por unidad de cómputo*, siempre que se valide el contenido —aquí la
   fluidez **ocultó un error conceptual**.
5. **Caveats de esta medición:** 1 sola muestra por modelo, VM compartida (otros becarios generaban carga concurrente),
   inferencia en **CPU**, modelos **pertenecientes a otros usuarios** (§4.4). No es un benchmark: es una prueba funcional
   de la API. Repetir aislado para sacar conclusiones de rendimiento.

---

## 6. Supuestos — estado tras el acceso real (19-oct-2026)

**Cerrados con medición en la VM:**

1. ~~VRAM FP16 sin medir~~ → **RESUELTO en la documentación, NO medido en la VM**: Google publica la tabla oficial de
   memoria (BF16 y Q4_0, §1.3) y el informe técnico añade el KV-cache. Lo que sigue sin medirse es la **ocupación real**
   en la VM: la Tesla T4 **no tiene driver cargado** (`nvidia-smi` falla, no hay `/dev/nvidia*`) y la inferencia va por
   **CPU** (8 vCPU). Pendiente de que un admin cargue el driver. Ver §4.2.
2. **VRAM Q4 = tamaños de artefactos GGUF publicados** → **parcialmente confirmado en la VM**: el artefacto `gemma4:e4b`
   desplegado es `parameter_size 7.5B`, `Q4_K_M`, 6.58 GB, ctx 131 072, lo que **cuadra con §1.3** (E4B ≈ 6.6 GB en Q4).
   Sigue sin medirse la ocupación real (GPU sin driver).
3. ~~Sin inferencia real~~ → **CERRADO**: el script se ejecutó 2 veces contra los dos endpoints de la VM (400 y 900 tokens
   de presupuesto). Respuestas + métricas en `results/comparativa_vm_20261019.json` y
   `results/comparativa_vm_20261019_budget900.json` (análisis en §5.1); el intento con el túnel caído quedó registrado en
   `results/fallo_tunel_ssh_20261019.json`.
4. ~~Asumida API compatible con OpenAI~~ → **CERRADO y corregido**: de las APIs descubiertas, **solo las de Ollama
   (11434/11435) son OpenAI-compatibles**; las de los puertos 8000/8001/18000 usan esquema propio (`{"prompt"}` o
   `?prompt=`) y la 18001 (`{"text","backend"}`) está **caída (502)**. El script soporta los 4 estilos (`openai|query|body|dual`).

**Siguen abiertos (requieren permisos o acceso que no tengo):**

5. **Gemma 4 no publica MMLU clásico ni HumanEval** (verificado en arXiv:2607.02770); no se incluyen cifras de esas
   métricas para Gemma 4. Cualquier valor de blogs comunitarios debe tratarse como **no oficial**.
6. **Nomenclatura:** las 5 variantes pedidas = familia **Gemma 4** (2026). Gemma 3n E2B/E4B (2025) reutiliza las mismas
   siglas para otra generación — confirmar en HF qué pesos se descargan.
7. **Idiomas** según documentación de lanzamiento (Gemma 4: 140+/35+; Qwen2.5: 29+; Llama 3: inglés; Phi-3 y Mistral 7B:
   inglés principalmente) — no se ejecutaron evals multilingües propias.
8. **Mistral 7B está "retired"** (docs.mistral.ai, 30-mar-2025): confirmar disponibilidad de pesos y evaluar el sucesor.
9. **Diferencias de harness entre benchmarks** (shots, CoT, versión): las cifras no son comparables entre modelos si
   provienen de informes distintos — reproducir con `lm-eval-harness` en la VM para una comparación homogénea.

**Nuevos, derivados de la VM compartida:**

10. **Los modelos de la VM no son míos**: `b_ruben_munoz_blanco` no tiene Ollama ni pesos propios; los endpoints usados
    (11434 `gemma4:e4b`, 11435 `gemma:latest`) pertenecen a **otros becarios** y se usan en modo solo-consulta. Sus APIs
    pueden caer o saturarse cuando ellos trabajan (los puertos 8000/8001/18000/18001 ya están caídos o colgados).
11. **Disco al 93 %** (38 GB libres de 497 GB): instalar modelos propios (>2-3 modelos grandes) debe dimensionarse antes.
12. **`gemma4:e4b` es un modelo de razonamiento**: devuelve `content` vacío y el texto en `reasoning` (en `/api/generate`
    la API lo llama `thinking`); el script tolera ese formato (si no, lo marcaría como fallo) y lo reporta con
    `reasoning_chars`/`used_reasoning_as_answer`.
13. **Ollama encola peticiones**: con la instancia de otro becario en uso, una llamada puede quedarse colgada (se observó
    un timeout de 180 s que desapareció al reintentar, 1,9 s después). Usar `--timeout` holgado y tratar los timeouts
    como **transitorios**, no como fallo definitivo de endpoint.
14. **Corrección importante (19-oct-2026):** la tabla de benchmarks que había antes tenía **las columnas E4B y 26B-A4B
    intercambiadas** y contenía dos filas (**SciCode** e **IFEval**) que **no están en la documentación oficial de Google**,
    además de valores inventados en CoVoST/FLEURS. Ya está corregido con los valores **verbatim** de la model card oficial
    (`huggingface.co/google/gemma-4-E4B-it/raw/main/README.md`). Lección: no mezclar tablas de terceros con la oficial.

---

## 7. Fuentes

| Fuente | Uso en este documento |
|---|---|
| `https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md` — model card cruda (verificación verbatim) | **Tabla de benchmarks y arquitectura (§1.4), corrección de columnas** |
| `https://ai.google.dev/gemma/docs/core` — tabla oficial de memoria BF16/Q4_0 | VRAM oficial (§1.3) |
| arXiv:2607.02770 — *Gemma 4 Technical Report* (v1 2-jul-2026, v2 24-jul-2026) | Arquitectura, tamaños, benchmarks (§1) |
| `https://huggingface.co/google/gemma-4-E4B-it` (y E2B / 12B / 26B-A4B / 31B) | Fichas de modelo, licencia Apache 2.0, modalidades (§1) |
| `https://developers.googleblog.com/gemma-4-12b-the-developer-guide` (3-jun-2026) | Fecha y arquitectura *encoder-free* del 12B (§1.2) |
| Blog oficial de Google "Gemma 4" y *Google Open Source* (2-abr-2026) | Fecha de lanzamiento de la familia (§1.2) |
| `https://ollama.com/library/gemma4` (y `llama3`, `phi3`, `mistral`) | Tamaños reales de artefactos Q4 (§1.3 y §2) |
| `https://ai.meta.com/blog/meta-llama-3/` + model card de Llama 3 | Datos de Llama 3 8B Instruct (§2) |
| arXiv:2404.14219 — *Phi-3 Technical Report* | Datos de Phi-3 mini (§2) |
| `https://docs.mistral.ai` — Mistral 7B Instruct v0.3 | Datos y estado "retired" (§2, §6) |
| `https://huggingface.co/learn/nlp-course` | Curso de NLP de Hugging Face (módulos introductorios) |
| **Medición propia en la VM (19-oct-2026)** | §4 completo y §5.1: hardware, APIs, latencias y respuestas |
