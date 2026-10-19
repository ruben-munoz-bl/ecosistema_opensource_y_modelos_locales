# Ecosistema OpenSource y Modelos Locales

Reto de autoformación: **comparativa de modelos LLM open-source** + **prueba de inferencia real** contra la VM de Azure de Inetum.

El entregable principal es [`comparativa.md`](comparativa.md): una comparativa de la familia **Gemma 4** (E2B, E4B, 12B, 26B-A4B, 31B) y de una selección de modelos de referencia (**Llama 3 8B**, **Phi-3 mini**, **Mistral 7B**), con datos contrastados contra fuentes oficiales, y un inventario medido in situ de la VM (hardware, APIs desplegadas, modelos disponibles y una ejecución real del mismo prompt en 2 modelos).

> **Estado:** completado el 19-oct-2026. Toda la evidencia (respuestas, latencias y métricas) está versionada en [`results/`](results/).

---

## Objetivo

El reto pide dos cosas:

1. **Comparar modelos open-source**: especificaciones, VRAM necesaria por cuantización, licencias, idiomas y benchmarks, con especial atención a la familia Gemma 4.
2. **Probarlo de verdad**: verificar qué modelos están desplegados en la VM de Azure compartida y ejecutar el **mismo prompt en al menos 2 modelos**, analizando la calidad de las respuestas.

La regla de oro del trabajo: **no mezclar fuentes**. Los datos de §1–§3 vienen de documentación oficial (fichas de modelo, arXiv, Ollama) y los de §4–§5 están **medidos en la VM**.

---

## Estructura del repositorio

```
.
├── comparativa.md        # Entregable principal (informe completo, §1–§7)
├── inference_azure.py    # Script de inferencia multi-estilo contra la VM
├── requirements.txt      # Dependencias del script (solo `requests`)
├── results/              # Evidencia JSON de las ejecuciones reales
│   ├── comparativa_vm_20261019.json              # Corrida A (max_tokens 400)
│   ├── comparativa_vm_20261019_budget900.json    # Corrida B (max_tokens 900)
│   └── fallo_tunel_ssh_20261019.json             # Fallo del túnel, registrado sin romper el script
└── .gitignore
```

### Cómo leer `comparativa.md`

| Sección | Contenido |
|---|---|
| **§1** | Familia Gemma 4: nomenclatura, especificaciones, modalidades, VRAM oficial y benchmarks |
| **§2** | Tabla comparativa general (5 modelos) + recomendación |
| **§3** | Análisis: qué modelo elegir con <16 GB de VRAM |
| **§4** | VM de Azure: acceso SSH, hardware medido, APIs descubiertas y modelos desplegados |
| **§5** | Uso de `inference_azure.py` y **análisis de calidad** de la ejecución real (§5.1) |
| **§6** | Supuestos: qué se cerró con evidencia y qué sigue abierto |
| **§7** | Fuentes |

---

## El script: `inference_azure.py`

Envía el **mismo prompt a 1–2 endpoints** de la VM, mide y guarda todo en `results/*.json`.

Soporta los **4 estilos de API** encontrados en la VM (se infieren de la URL o se fuerzan con `--api`):

| Estilo | Endpoint típico | Petición |
|---|---|---|
| `openai` | `/v1/chat/completions` | JSON compatible OpenAI (el que usan los Ollama) |
| `query` | `/chat?prompt=...` | prompt como parámetro de query |
| `body` | `/chat` | `{"prompt": "..."}` |
| `dual` | `/api/chat` | `{"prompt": "..."}` → `{"text", "backend"}` |

**Métricas registradas:** latencia, tokens de entrada/salida (de la API o estimados), tokens/s, `finish_reason` (`length` = respuesta cortada), `backend`, y `reasoning_chars` / `used_reasoning_as_answer` para modelos de razonamiento cuyo texto sale en el campo `reasoning`.

**Códigos de salida:** `0` = todo OK · `1` = algún destino falló (el JSON se guarda igual) · `2` = configuración inválida.

### Uso básico

```powershell
# 0) Túnel SSH (las APIs viven en 127.0.0.1 de la VM) — terminal aparte
ssh -N -L 11434:127.0.0.1:11434 -L 11435:127.0.0.1:11435 vm-b_ruben_munoz_blanco

# 1) Ejecutar el mismo prompt en 2 modelos
python inference_azure.py `
  --service "gemma4:e4b=http://127.0.0.1:11434/v1/chat/completions" `
  --service "gemma:latest=http://127.0.0.1:11435/v1/chat/completions" `
  --prompt "Resume en 3 frases qué es un modelo de mezcla de expertos (MoE)." `
  --max-tokens 900 --timeout 900
```

Modo legado compatible OpenAI (`AZURE_VM_URL` + `--model`), otros ejemplos y todas las opciones: ver **§5** de [`comparativa.md`](comparativa.md).

---

## Resultados destacados (§5.1)

Mismo prompt, temperatura 0.2, en los dos endpoints que respondían de verdad:

| Corrida | Modelo | Latencia | Tokens out | `finish_reason` |
|---|---|---:|---:|---|
| B (900) | `gemma4:e4b` (7,5B · Q4_K_M) | **249,9 s** | 763 | `stop` |
| B (900) | `gemma:latest` (9B · Q4_0) | **20,7 s** | 64 | `stop` |

- **Calidad de contenido: gana `gemma4:e4b`.** Explica MoE correctamente (expertos + *gating* + activación dispersa) y lo conecta con la inferencia local. `gemma:latest` confunde MoE con un *ensemble* y "conocimiento local" — fluido, pero **conceptualmente incorrecto**.
- **Coste: gana `gemma:latest` por 12×.** En CPU sin driver de GPU, el modelo razonador paga un peaje grande de latencia.
- **Lección práctica:** para modelos de razonamiento, presupuesto de tokens ≥ 800–1000 (con 400 se cortó sin llegar a responder) y **validar el contenido** — la fluidez puede ocultar errores.

---

## Notas de rigor

- **VM compartida, uso solo-lectura:** todo se ejecutó con el usuario `b_ruben_munoz_blanco`, sin `sudo` y sin tocar procesos, ficheros ni servicios de otros becarios. Los modelos de los puertos 11434/11435 **pertenecen a otros usuarios** y se consultaron en modo solo-consulta.
- **La GPU (Tesla T4) no tiene driver cargado**, así que la inferencia medida va por **CPU** y la VRAM real no es medible.
- **No es un benchmark:** 1 muestra por modelo, VM compartida con carga concurrente. Es una prueba funcional de las APIs.
- **Datos verificados:** los benchmarks de Gemma 4 se contrastaron *verbatim* con la model card oficial (esto corrigió columnas intercambiadas y filas no oficiales); las cifras de memoria son las de la tabla oficial de Google (BF16/Q4_0) más los tamaños reales de los GGUF publicados.

## Fuentes principales

Ficha oficial de Gemma 4 (`ai.google.dev/gemma/docs/core`), *Gemma 4 Technical Report* (arXiv:2607.02770), model cards de Hugging Face, biblioteca de Ollama, documentación de Meta/Microsoft/Mistral. Lista completa con su uso concreto en **§7** de [`comparativa.md`](comparativa.md).
