#!/usr/bin/env python3
"""Compara la misma respuesta en 1 o 2 modelos a traves de la API REST
compatible con OpenAI de la VM de Azure (sin descargar pesos ni tocar HF).

Requiere la variable de entorno AZURE_VM_URL (p. ej. http://<ip-vm>:8000).

Uso:
    export AZURE_VM_URL="http://<ip-vm>:8000"
    python inference_azure.py --model gemma4:12b llama3:8b --prompt "Explica que es un MoE."
    python inference_azure.py --model qwen2.5:14b --prompt-file prompt.txt --max-tokens 512

Salida: tabla resumen por consola + JSON en results/results_<timestamp>.json
Metricas por modelo: latencia (s), tokens (usage de la API o estimacion chars/4),
longitud (chars/lineas) y tokens/s aproximados.

Codigos de salida: 0 = ok total, 1 = fallo en al menos un modelo, 2 = configuracion invalida.
"""

import argparse
import datetime as _dt
import json
import os
import sys
import time

try:
    import requests
except ImportError:  # pragma: no cover
    print("Falta la dependencia 'requests'. Ejecuta: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(2)

DEFAULT_PATH = "/v1/chat/completions"
DEFAULT_PROMPT = "Explica en 3 frases que es un modelo de mezcla de expertos (MoE)."


def approx_tokens(text: str) -> int:
    """Estimacion grosera de tokens (~4 caracteres por token en ingles/esp.).
    Solo se usa cuando la API no devuelve el campo 'usage'."""
    return max(1, round(len(text) / 4))


def extract_content(data: dict) -> str:
    """Extrae el texto de una respuesta con formato OpenAI; tolera variaciones."""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("respuesta sin 'choices'")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):  # formato multimodal [{type, text}]
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        content = str(content)
    if not content.strip():
        raise ValueError("respuesta con contenido vacio")
    return content


def run_model(base_url: str, path: str, model: str, messages: list, args) -> dict:
    """Ejecuta un prompt contra un modelo. Devuelve metricas + respuesta.
    Lanza excepcion (requests/ValueError) que captura el llamante."""
    url = base_url.rstrip("/") + path
    payload = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=args.timeout)
    elapsed = time.perf_counter() - started
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    content = extract_content(data)

    usage = data.get("usage") or {}
    tokens_out = usage.get("completion_tokens")
    tokens_in = usage.get("prompt_tokens")
    source = "api"
    if not isinstance(tokens_out, int):
        tokens_out = approx_tokens(content)
        source = "estimado (chars/4)"
    if not isinstance(tokens_in, int):
        tokens_in = approx_tokens(" ".join(m["content"] for m in messages))

    return {
        "model": model,
        "ok": True,
        "latency_s": round(elapsed, 3),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_source": source,
        "tokens_per_s": round(tokens_out / elapsed, 2) if elapsed > 0 else None,
        "chars": len(content),
        "lines": content.count("\n") + 1,
        "response": content,
        "error": None,
    }


def run_model_safe(base_url: str, path: str, model: str, messages: list, args) -> dict:
    try:
        return run_model(base_url, path, model, messages, args)
    except requests.exceptions.Timeout:
        err = f"timeout tras {args.timeout}s"
    except requests.exceptions.ConnectionError as exc:
        err = f"error de conexion a {base_url}: {exc.__class__.__name__}"
    except (ValueError, KeyError) as exc:
        err = f"respuesta malformada: {exc}"
    except RuntimeError as exc:
        err = str(exc)
    except Exception as exc:  # noqa: BLE001 - no abortar la corrida completa
        err = f"{exc.__class__.__name__}: {exc}"
    print(f"  [FALLO] {model}: {err}", file=sys.stderr)
    return {
        "model": model,
        "ok": False,
        "latency_s": None,
        "tokens_in": None,
        "tokens_out": None,
        "tokens_source": None,
        "tokens_per_s": None,
        "chars": None,
        "lines": None,
        "response": None,
        "error": err,
    }


def print_summary(results: list) -> None:
    header = f"{'MODELO':<28} {'OK':<4} {'LAT (s)':>9} {'TOK OUT':>9} {'TOK/S':>8} {'CHARS':>7} {'FUENTE':>16}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<28} "
            f"{('si' if r['ok'] else 'NO'):<4} "
            f"{(r['latency_s'] if r['latency_s'] is not None else '-'):>9} "
            f"{(r['tokens_out'] if r['tokens_out'] is not None else '-'):>9} "
            f"{(r['tokens_per_s'] if r['tokens_per_s'] is not None else '-'):>8} "
            f"{(r['chars'] if r['chars'] is not None else '-'):>7} "
            f"{(r['tokens_source'] or '-'):>16}"
        )


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ejecuta el mismo prompt en 1-2 modelos via REST y guarda respuestas + metricas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        nargs="+",
        required=True,
        metavar="NOMBRE",
        help="uno o dos nombres de modelo (p. ej. gemma4:12b llama3:8b)",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt inline")
    p.add_argument("--prompt-file", help="lee el prompt de un archivo (tiene prioridad sobre --prompt)")
    p.add_argument("--system", default=None, help="mensaje de sistema opcional")
    p.add_argument("--max-tokens", type=int, default=512, help="maximo de tokens de salida")
    p.add_argument("--temperature", type=float, default=0.2, help="temperatura de muestreo")
    p.add_argument("--timeout", type=float, default=120.0, help="timeout HTTP en segundos")
    p.add_argument("--path", default=DEFAULT_PATH, help="ruta del endpoint relativa a AZURE_VM_URL")
    p.add_argument("--out", default=None, help="ruta del JSON de salida (por defecto results/results_<ts>.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if len(args.model) > 2:
        print("ERROR: --model admite como maximo 2 modelos (comparacion A vs B).", file=sys.stderr)
        return 2

    base_url = os.environ.get("AZURE_VM_URL", "").strip()
    if not base_url:
        print(
            "ERROR: falta la variable de entorno AZURE_VM_URL.\n"
            '  PowerShell: $env:AZURE_VM_URL="http://<ip-vm>:8000"\n'
            '  bash:       export AZURE_VM_URL="http://<ip-vm>:8000"',
            file=sys.stderr,
        )
        return 2

    prompt = args.prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as fh:
                prompt = fh.read().strip()
        except OSError as exc:
            print(f"ERROR: no se pudo leer --prompt-file: {exc}", file=sys.stderr)
            return 2
        if not prompt:
            print("ERROR: el archivo de prompt esta vacio.", file=sys.stderr)
            return 2

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    print(f"Endpoint : {base_url.rstrip('/')}{args.path}")
    print(f"Modelos  : {', '.join(args.model)}")
    print(f"Prompt   : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

    results = []
    for i, model in enumerate(args.model, 1):
        print(f"[{i}/{len(args.model)}] ejecutando '{model}' ...")
        results.append(run_model_safe(base_url, args.path, model, messages, args))

    print_summary(results)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("results", f"results_{stamp}.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "endpoint": base_url.rstrip("/") + args.path,
        "prompt": prompt,
        "params": {"max_tokens": args.max_tokens, "temperature": args.temperature, "timeout": args.timeout},
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en: {out_path}")

    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
