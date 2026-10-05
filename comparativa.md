# Reto del Día — Comparativa de modelos LLM open-source

> **Fecha de elaboración:** 05-oct-2026 · **Restricción:** sin acceso a Hugging Face Hub ni a la VM de Azure.
> Todo dato procede de **documentación oficial pública** (fichas de modelo, informes técnicos arXiv, documentación de
> proveedores) o se marca explícitamente como **ESTIMACIÓN**. Las fuentes se listan al final.

---

## 1. Familia Gemma — variantes E2B, E4B, 12B, 26B-A4B, 31B

### 1.1 Nota de nomenclatura (obligatoria)

**Las 5 variantes existen públicamente a fecha Oct-2026.** Corresponden a la familia **Gemma 4** de Google DeepMind:

| Lanzamiento | Variantes |
|---|---|
| 31-mar / 2-abr 2026 | E2B, E4B, 31B, 26B A4B (+ MTP el 16-abr 2026) |
| 3-jun 2026 | 12B Unified |

⚠️ **Posible confusión de nomenclatura:** las siglas **E2B/E4B también existen en Gemma 3n** (generación anterior, mayo-2025).
Aquí se analiza **Gemma 4** (2026), que es la única familia que contiene las 5 variantes pedidas simultáneamente
(E2B + E4B + 12B + 26B-A4B + 31B). No hay ninguna variante "no encontrada".

**Licencia de toda la familia: Apache 2.0.** · Idiomas: preentrenada en **140+ idiomas**, soporte "out-of-the-box" en **35+ idiomas**.

### 1.2 Especificaciones y modalidades (fuente: ficha oficial `ai.google.dev/gemma/docs/core/model_card_4` + informe técnico arXiv:2607.02770)

| Variante | Parámetros efectivos | Parámetros totales | Capas | Contexto | Modalidades (entrada → salida) | Encoders |
|---|---|---|---|---|---|---|
| **Gemma 4 E2B** | **2.3B** (2,340M núcleo) | 5.1B con embeddings | 35 | 128K | Texto, Imagen, **Audio** → Texto | visión ~150M + audio ~305M |
| **Gemma 4 E4B** | **4.5B** | 8B con embeddings | 42 | 128K | Texto, Imagen, **Audio** → Texto | visión ~150M + audio ~305M |
| **Gemma 4 12B Unified** | **11.95B** (sin encoders: arquitectura *encoder-free*) | 11.95B + ~550M auxiliares | 48 | 256K | Texto, Imagen, **Audio** → Texto | sin encoders (proyecta parches de imagen y ondas de audio directamente) |
| **Gemma 4 26B A4B** | **3.8B activos** (MoE: 8 experts activos / 128 totales + 1 compartido) | **25.2B** | 30 | 256K | Texto, Imagen → Texto | visión ~550M · **sin audio** |
| **Gemma 4 31B** (denso) | **30.7B** (denso, todos activos) | 30.7B + ~450M auxiliares | 60 | 256K | Texto, Imagen → Texto | visión ~550M · **sin audio** |

Notas:
- **E2B/E4B**: el prefijo "E" = *effective parameters*. Gracias a MatFormer + Per-Layer Embeddings (PLE), solo ~2B/~4B residen en el acelerador; los embeddings por capa pueden descargarse en RAM de CPU.
- **26B A4B**: la "A" = *Active parameters*; velocidad de inferencia cercana a un 4B con calidad cercana al 31B.
- **12B Unified**: único modelo *encoder-free* de la familia (todo fluye por un solo decoder-only).

### 1.3 VRAM estimada (FP16) y real en GGUF Q4

**Fórmula FP16 (ESTIMACIÓN):** `pesos × 2 bytes + encoders auxiliares + KV cache @ ~1K tokens`.
Las cifras FP16 coinciden con las calculadas por apxml (tercero) para 1.024 tokens; **excluyen** crecimiento de KV con contexto largo.

| Variante | Pesos (B) | VRAM FP16 @1K tok (ESTIMACIÓN) | VRAM Q4 — artefacto GGUF publicado (dato real, Ollama) |
|---|---|---|---|
| E2B | 5.1 | **≈ 11.5 GB** | **4.6 – 7.5 GB** (según cuantización) |
| E4B | 8.0 | **≈ 17.2 GB** | **6.6 – 9.5 GB** |
| 12B Unified | 11.95 | **≈ 25.3 GB** | **7.7 – 8.0 GB** |
| 26B A4B | 25.2 (3.8 activos) | **≈ 51.0 GB** (todos los experts cargados) | **16 – 19 GB** |
| 31B | 30.7 | **≈ 62.5 GB** | **19 – 20 GB** |

Regla práctica: **VRAM Q4 ≈ pesos × 0.55–0.65 bytes/param** + KV cache. Con E2B/E4B + PLE offload, la VRAM *real del acelerador* puede bajar hasta ~4.6/~9 GB en FP16 del núcleo (ESTIMACIÓN; verificar con `nvidia-smi` en la VM).

### 1.4 Benchmarks públicos (HECHOS — tabla oficial, modelos *instruction-tuned*)

Fuente: ficha oficial del modelo Gemma 4 y arXiv:2607.02770. **Google NO publica "MMLU" clásico ni "HumanEval" para Gemma 4** (ver §1.5):

| Benchmark | 31B | 26B A4B | 12B | E4B | E2B |
|---|---:|---:|---:|---:|---:|
| MMLU-Pro (%) | **85.2** | 82.6 | 77.2 | 69.4 | 60.0 |
| GPQA Diamond (%) | 84.3 | 82.3 | 78.8 | 58.6 | 43.4 |
| AIME 2026 no tools (%) | 89.2 | 88.3 | 77.5 | 42.5 | 37.5 |
| LiveCodeBench v6 (%) | 80.0 | 77.1 | 72.0 | 52.0 | 44.0 |
| SciCode (%) | 43.0 | 40.0 | 38.0 | 24.0 | 21.0 |
| Codeforces ELO | 2150 | 1718 | 1659 | 940 | 633 |
| BigBench Extra Hard (%) | 74.4 | 64.8 | 53.0 | 33.1 | 21.9 |
| MMMLU, multilingüe (%) | 88.4 | 86.3 | 83.4 | 76.6 | 67.4 |
| IFEval (%) | 98.9 | 98.5 | 97.2 | 96.7 | 94.6 |
| HLE sin herramientas (%) | 19.5 | 8.7 | 5.2 | – | – |
| Tau2, media de 3 (%) | 76.9 | 68.2 | 69.0 | 42.2 | 24.5 |
| MMMU-Pro, visión (%) | 76.9 | 73.8 | 69.1 | 52.6 | 44.2 |
| MATH-Vision (%) | 85.6 | 82.4 | 79.7 | 59.5 | 52.4 |
| MRCR v2 8-aguja @128K (%) | 66.4 | 44.1 | 43.4 | 25.4 | 19.1 |
| CoVoST, audio (BLEU) | – | – | sí (informe, Tabla 8) | 35.54 | 33.47 |
| FLEURS, audio (WER, menor=mejor) | – | – | sí (informe) | 0.08 | 0.09 |

### 1.5 Sobre MMLU y HumanEval en Gemma 4 — declaración explícita

- **MMLU (clásico) y HumanEval: NO publicados oficialmente** para Gemma 4. Verifiqué el informe técnico completo (arXiv:2607.02770): no contienen esas métricas. **No se inventan cifras.**
- **Alternativas oficiales equivalentes** que sí existen: **MMLU-Pro** (conocimiento), **LiveCodeBench v6 / SciCode / Codeforces ELO** (código).
- Existen cifras no oficiales circulando (p. ej. blog comunitario `gemma4-ai.com`: MMLU 87.1 / HumanEval 76.8 para 31B), pero **se descartan**: fuente sin trazabilidad y contradice la ficha oficial (ese blog afirma que el 31B es "texto solo", cuando la ficha oficial incluye MMMU-Pro 76.9 para 31B). **No se usan.**

---

## 2. Tabla comparativa general (5 modelos)

VRAM: **FP16 = ESTIMACIÓN** (`pesos × 2 bytes`, sin KV largo) · **Q4 = tamaño de artefacto publicado** (GGUF de Ollama, dato real, q4_K_M/q4_0).

| Modelo | Parámetros | VRAM FP16 (est.) | VRAM Q4 (real) | Licencia | Idiomas | Modalidades | Casos de uso ideal |
|---|---|---|---|---|---|---|---|
| **Gemma 4 12B Unified** (Google, 2026) | 11.95B | ≈ 25 GB | **7.7 – 8.0 GB** | Apache 2.0 | 140+ preentreno / 35+ soportados | Texto+Imagen+Audio → Texto | Asistente on-prem multimodal (docs+voz+imagen), RAG con 256K ctx, fine-tuning en estación de trabajo |
| **Llama 3 8B Instruct** (Meta, 2024) | 8.03B | ≈ 16 GB | **4.7 GB** (Q4_0) | Meta Llama 3 Community License (uso comercial permitido con cláusulas; >700M MAU requiere licencia) | Inglés (Llama 3.1: 8 idiomas) | Texto → Texto | Chatbot general, RAG, base para SFT/LoRA; ecosistema y tooling más maduro |
| **Phi-3-mini-4k-instruct** (Microsoft, 2024) | 3.82B | ≈ 7.6 GB | **2.2 GB** (Q4_0) | MIT | Inglés (principalmente) | Texto → Texto | Edge/móvil, latencia mínima, SLM razonador, copiloto embebido |
| **Mistral 7B Instruct v0.3** (Mistral, 2024) | 7.25B | ≈ 14.6 GB (docs Mistral: ≈20 GB bf16 con contexto) | **4.4 GB** (Q4_K_M) | Apache 2.0 | Inglés (principal) | Texto → Texto | Assistant general, function calling (v0.3), despliegue económico. *Modelo retirado 30-mar-2025 → sucesor Ministral 3 8B* |
| **Qwen2.5-14B-Instruct** (Alibaba, 2024) | 14.7B | ≈ 29 GB | **9.0 GB** (Q4_K_M) | Apache 2.0 (los tamaños 3B y 72B usan licencia Qwen) | **29+ idiomas** (EN, ZH, ES, FR, DE, PT, RU, JA, KO, AR…) | Texto → Texto | Multilingüe (ES incluido), salida estructurada/JSON, código, RAG 128K ctx |

### Benchmarks de referencia (solo fuentes oficiales, setups distintos → no directamente comparables entre filas)

| Modelo | MMLU | HumanEval | Fuente |
|---|---:|---:|---|
| Gemma 4 12B | *n/d* → **MMLU-Pro 77.2** | *n/d* → **LiveCodeBench v6 72.0** | Ficha oficial Gemma 4 |
| Llama 3 8B Instruct | 68.4 (5-shot) | 62.2 (0-shot) | Meta `MODEL_CARD.md` |
| Phi-3-mini (4k) | 68.8 (5-shot) | 58.5 (0-shot) | Informe técnico arXiv:2404.14219 |
| Mistral 7B Instruct v0.3 | 60.1 (5-shot, blog Mistral) / 61.8 (re-eval OpenLLM) | ≈ 28 (0-shot, tabla del informe Phi-3 sobre Mistral 7B) | mistral.ai / RedHatAI |
| Qwen2.5-14B (base) | 79.7 | 57.9 (Qwen2.5-**7B** base) | Informe técnico arXiv:2412.15115 |

---

## 3. Análisis — ¿qué modelo elegir para producción con <16 GB de VRAM?

1. **Mi elección: Gemma 4 12B Unified en Q4 (~7.7–8.0 GB)** — es el mejor ratio calidad/VRAM de la tabla: MMLU-Pro 77.2 y GPQA-D 78.8 (oficiales) con calidad de modelo ~4× más grande, deja ~8 GB de margen en una GPU de 16 GB para KV cache y activaciones con ventanas largas, licencia Apache 2.0 sin cláusulas y modalidades texto+imagen+audio sin coste añadido.
2. **Alternativa multilingüe: Qwen2.5-14B Q4 (~9.0 GB)** — si el producto es 100% texto en español/idiomas no ingleses y requiere JSON estructurado: 29+ idiomas y Apache 2.0; paga el peaje de ~1 GB extra de VRAM y sin modalidades.
3. **Si manda latencia/coste: Llama 3.1 8B Q4 (~4.7 GB) o Phi-3-mini (~2.2 GB)** — caben incluso en GPUs de 8/4 GB y dejan margen para batch, pero su techo de calidad (MMLU 68.4 / 68.8) queda por debajo del 12B (MMLU-Pro 77.2, métrica distinta: no comparables 1:1).
4. **Descartados para 16 GB:** Gemma 4 26B A4B (Q4 16–19 GB, se come toda la VRAM sin margen para KV) y Gemma 4 31B (Q4 19–20 GB → no cabe); el Mistral 7B, aunque ligero, está retirado desde 2025 y su calidad está por debajo de la generación 2026.
5. **Caveat de ingeniería:** las cifras FP16 son estimaciones y las Q4 son tamaños de artefactos publicados; la VRAM real (KV + activaciones + fragmentación) debe medirse con `nvidia-smi` en la VM de Azure antes de fijar el modelo en producción.

---

## 4. `inference_azure.py` — instrucciones de uso

El script está **listo pero NO ejecutado** (la VM no está disponible). Usa solo `requests` contra la API REST
compatible con OpenAI de la VM. La URL base se toma de la variable de entorno **`AZURE_VM_URL`**.

```bash
# 1) Preparar entorno (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2) Configurar la URL de la VM (ajusta IP/puerto cuando vuelva a existir)
$env:AZURE_VM_URL = "http://<IP_VM>:8000"     # bash: export AZURE_VM_URL="http://<IP_VM>:8000"

# 3) Comparar DOS modelos con el MISMO prompt (guarda respuestas + métricas en results/)
python inference_azure.py --model gemma4:12b llama3:8b --prompt "Resume en 3 frases qué es un MoE."

# 4) Un solo modelo, prompt desde archivo, parámetros ajustados
python inference_azure.py --model qwen2.5:14b --prompt-file prompt.txt --max-tokens 512 --temperature 0.2

# 5) Endpoint o timeout distintos (si la VM no expone /v1/chat/completions)
python inference_azure.py --model gemma4:12b phi3:mini --path /v1/chat/completions --timeout 180

# 6) Salida explícita
python inference_azure.py --model gemma4:12b llama3:8b --out results/comparativa_01.json
```

**Métricas registradas por modelo:** latencia wall-time (s), tokens de entrada/salida (del campo `usage` de la API si existe; si no, estimación `caracteres/4`), longitud en caracteres/líneas y tokens/s aproximados. Todo se imprime como tabla resumen y se persiste en `results/results_<timestamp>.json`.

**Códigos de salida:** `0` = todos los modelos respondieron · `1` = al menos un modelo falló · `2` = falta `AZURE_VM_URL` o argumentos inválidos.

---

## 5. Supuestos por falta de acceso (deben verificarse en HF/Azure cuando vuelva el acceso)

1. **VRAM FP16 de la tabla §1.3 y §2 = ESTIMACIÓN** (`pesos×2 + encoders + KV @1K tok`), coherente con apxml pero **sin medición en la VM**; verificar con `nvidia-smi` bajo carga real y con las cuantizaciones efectivamente desplegadas.
2. **VRAM Q4 = tamaños de artefactos GGUF publicados en Ollama** (proxy de terceros): las cuantizaciones de HF (AWQ/GPTQ/bitsandbytes) pueden variar ±10–15 %. Verificar en HF cuando vuelva el acceso.
3. **Gemma 4 no publica MMLU clásico ni HumanEval** (verificado en arXiv:2607.02770); no se incluyen cifras de esas métricas para Gemma 4. Cualquier valor que aparezca en blogs comunitarios debe tratarse como **no oficial**.
4. **Nomenclatura:** las 5 variantes pedidas = familia **Gemma 4** (2026). Gemma 3n E2B/E4B (2025) reutiliza las mismas siglas para otra generación — confirmar en HF qué pesos se descargan.
5. **Idiomas** según documentación de lanzamiento (Gemma 4: 140+/35+; Qwen2.5: 29+; Llama 3: inglés; Phi-3 y Mistral 7B: inglés principalmente) — no se ejecutaron evals multilingües propias.
6. **Sin inferencia real:** las métricas de latencia/tokens/s del script se generarán en la VM; hasta entonces el script solo está verificado sintácticamente (`py_compile`).
7. **Asumida API compatible con OpenAI (`/v1/chat/completions`)**, válida para vLLM/Ollama/LiteLLM; si la VM expone un esquema propio, ajustar `--path` y el payload.
8. **Mistral 7B está "retired"** (docs.mistral.ai, 30-mar-2025, reemplazado por Ministral 3 8B): confirmar disponibilidad de pesos y conviene evaluar el sucesor.
9. **Diferencias de harness entre benchmarks** (shots, CoT, versión): las cifras no son comparables entre modelos si provienen de informes distintos — reproducir con `lm-eval-harness` en la VM para una comparación homogénea.

---

## 6. Fuentes

| Fuente | Uso |
|---|---|
| `https://ai.google.dev/gemma/docs/core/model_card_4` (ficha oficial Gemma 4) | Especificaciones, modalidades, benchmarks §1 |
| `https://arxiv.org/abs/2607.02770` — *Gemma 4 Technical Report* | Parámetros por componente, confirmación de ausencia de MMLU/HumanEval, audio del 12B |
| `https://ollama.com/library/gemma4` (+ tags `e2b`, `e4b`, `12b`, `26b`, `31b`) | Tamaños GGUF Q4 reales, espejo de la tabla oficial |
| `https://apxml.com/models/gemma-4-{e2b,e4b,12b,26b-a4b,31b}` | VRAM FP16 por variante (tercero) |
| `https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md` | Llama 3: MMLU/HumanEval/GSM8K, licencia |
| `https://arxiv.org/abs/2404.14219` — *Phi-3 Technical Report* | Phi-3-mini/small/medium: MMLU, HumanEval, MT-Bench |
| `https://docs.mistral.ai/models/mistral-7b-0-3` y `https://mistral.ai/news/announcing-mistral-7b` | Mistral 7B: licencia, contexto 32K, RAM oficial 20→5 GB, retiro del modelo |
| `https://arxiv.org/abs/2412.15115` — *Qwen2.5 Technical Report* | Qwen2.5: MMLU, licencias por tamaño, 29+ idiomas |
| Tags de Ollama `llama3:8b`, `phi3:mini`, `mistral:7b`, `qwen2.5:14b` | Tamaños Q4 reales de la tabla §2 |
