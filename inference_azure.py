#!/usr/bin/env python3
"""Compara la misma respuesta en 1-2 modelos/servicios contra las APIs REST de la VM de Azure.

Estilos soportados (--api), segun los openapi.json descubiertos en la VM (ver comparativa.md 4.3):

    openai  POST <base>/v1/chat/completions  {"model","messages",...} -> choices[0].message.content
    query   POST <base>/chat?prompt=<texto>                            -> texto (puertos 8000, 18000)
    body    POST <base>/chat          {"prompt": "..."}                -> texto (puerto 8001)
    dual    POST <base>/api/chat      {"prompt": "..."}                -> {"text","backend"} (puerto 18001)

CONECTIVIDAD: las APIs escuchan en 127.0.0.1 de la VM y NO estan publicadas a internet.
Se accede mediante tunel SSH (desde otra terminal/pestania):
    ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 -L 18001:127.0.0.1:18001 vm-b_ruben_munoz_blanco
y luego:
    export AZURE_VM_URL="http://127.0.0.1:8000"   # PowerShell: $env:AZURE_VM_URL="http://127.0.0.1:8000"

Uso:
    # A) Estilo legado compatible con OpenAI: --model (1-2 nombres) contra AZURE_VM_URL + --path
    export AZURE_VM_URL="http://127.0.0.1:11440"
    python inference_azure.py --model gemma4:e4b gemma:latest --prompt "Explica que es un MoE."

    # B) APIs propias de la VM: --service NOMBRE=URL (1-2), mismo prompt para todas
    python inference_azure.py --api dual --service dual=http://127.0.0.1:18001/api/chat
    python inference_azure.py --api body --service a=http://127.0.0.1:8001/chat --service b=http://127.0.0.1:8000/chat
    python inference_azure.py --api query --service api8000=http://127.0.0.1:8000/chat --prompt-file prompt.txt

Salida: tabla resumen por consola + JSON en results/results_<timestamp>.json
Metricas por servicio: latencia (s), tokens (usage de la API o estimacion chars/4),
longitud (chars/lineas), tokens/s aproximados y campo 'backend' si la API lo devuelve.

Codigos de salida: 0 = ok total, 1 = fallo en al menos un servicio, 2 = configuracion invalida.
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
API_STYLES = ("openai", "query", "body", "dual")


def approx_tokens(text: str) -> int:
    """Estimacion grosera de tokens (~4 caracteres por token en ingles/esp.).
    Solo se usa cuando la API no devuelve el campo 'usage'."""
    return max(1, round(len(text) / 4))


def extract_content(data: dict) -> str:
    """Extrae el texto de una respuesta con formato OpenAI; tolera variaciones.
    Modelos de razonamiento (p. ej. gemma4:e4b en Ollama) devuelven 'content' vacio
    y el texto en 'reasoning'; en ese caso se usa 'reasoning' como respuesta."""
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("respuesta sin 'choices'")
    message = choices[0].get("message") or {}

    def _flatten(value):
        if isinstance(value, list):  # formato multimodal [{type, text}]
            value = "".join(part.get("text", "") for part in value if isinstance(part, dict))
        return value if isinstance(value, str) else str(value or "")

    content = _flatten(message.get("content", "")).strip()
    reasoning = _flatten(message.get("reasoning", "")).strip()
    if content:
        return content
    if reasoning:
        return reasoning
    raise ValueError(
        "respuesta con contenido vacio"
        + (" (tenia 'reasoning' pero vacio)" if reasoning is not None else "")
    )


def extract_plain_content(data: dict) -> str:
    """Extrae texto de las APIs propias de la VM (8000/8001/18000/18001).
    Acepta las formas habituales: {'text'}, {'response'}, {'message'}, {'content'} o texto plano."""
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        raise ValueError(f"respuesta de tipo inesperado: {type(data).__name__}")
    for key in ("text", "response", "message", "content", "answer", "result"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):  # [{type: 'text', text: '...'}]
            joined = "".join(
                part.get("text", "") for part in value if isinstance(part, dict)
            ).strip()
            if joined:
                return joined
    # Ultimo recurso: si solo hay un campo escalar, devolverlo; si no, error con lo recibido
    scalars = {k: v for k, v in data.items() if isinstance(v, str) and v.strip()}
    if len(scalars) == 1:
        return next(iter(scalars.values()))
    raise ValueError(f"no se localizo el texto en la respuesta (claves: {sorted(data)})")


def build_call(style: str, url: str, model, messages: list, args):
    """Devuelve (kwargs para requests.post) segun el estilo de API."""
    if style == "openai":
        payload = {
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        if model:
            payload["model"] = model
        return {"json": payload}

    # Los estilos propios de la VM solo aceptan un prompt suelto: usamos el ultimo mensaje de usuario
    prompt = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        messages[-1]["content"] if messages else "",
    )
    if style == "query":
        return {"params": {"prompt": prompt}}
    if args.system:  # body/dual no tienen campo de sistema: se antepone al prompt
        prompt = f"{args.system}\n\n{prompt}"
    return {"json": {"prompt": prompt}}


def parse_response(style: str, data: dict) -> tuple:
    """Devuelve (texto, backend, reasoning). reasoning solo existe en estilos con 'message'
    (modelos de razonamiento tipo gemma4:e4b); en el resto es None."""
    if style == "openai":
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        reasoning = message.get("reasoning")
        reasoning = reasoning if isinstance(reasoning, str) and reasoning.strip() else None
        return extract_content(data), data.get("backend"), reasoning
    backend = data.get("backend") if isinstance(data, dict) else None
    return extract_plain_content(data), (backend if isinstance(backend, str) else None), None


def run_model(target: dict, messages: list, args) -> dict:
    """Ejecuta un prompt contra un destino {name, url, style, model}. Devuelve metricas + respuesta.
    Lanza excepcion (requests/ValueError) que captura el llamante."""
    url, style = target["url"], target["style"]
    started = time.perf_counter()
    resp = requests.post(url, timeout=args.timeout, **build_call(style, url, target.get("model"), messages, args))
    elapsed = time.perf_counter() - started
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    content, backend, reasoning = parse_response(style, data)

    # finish_reason indica si la respuesta se corto por max_tokens ("length") o termino ("stop")
    finish_reason = None
    if isinstance(data, dict) and data.get("choices"):
        finish_reason = data["choices"][0].get("finish_reason")

    usage = data.get("usage") if isinstance(data, dict) else None
    usage = usage or {}
    tokens_out = usage.get("completion_tokens")
    tokens_in = usage.get("prompt_tokens")
    source = "api"
    if not isinstance(tokens_out, int):
        tokens_out = approx_tokens(content)
        source = "estimado (chars/4)"
    if not isinstance(tokens_in, int):
        tokens_in = approx_tokens(" ".join(m["content"] for m in messages))

    return {
        "model": target["name"],
        "url": url,
        "api_style": style,
        "ok": True,
        "latency_s": round(elapsed, 3),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_source": source,
        "tokens_per_s": round(tokens_out / elapsed, 2) if elapsed > 0 else None,
        "chars": len(content),
        "lines": content.count("\n") + 1,
        "backend": backend,
        "finish_reason": finish_reason,
        "reasoning_chars": len(reasoning) if reasoning else 0,
        "used_reasoning_as_answer": bool(reasoning) and content.strip() == reasoning.strip(),
        "response": content,
        "error": None,
    }


def run_model_safe(target: dict, messages: list, args) -> dict:
    name = target["name"]
    url = target["url"]
    try:
        return run_model(target, messages, args)
    except requests.exceptions.Timeout:
        err = f"timeout tras {args.timeout}s"
    except requests.exceptions.ConnectionError as exc:
        err = f"error de conexion a {url}: {exc.__class__.__name__}"
    except (ValueError, KeyError) as exc:
        err = f"respuesta malformada: {exc}"
    except RuntimeError as exc:
        err = str(exc)
    except Exception as exc:  # noqa: BLE001 - no abortar la corrida completa
        err = f"{exc.__class__.__name__}: {exc}"
    print(f"  [FALLO] {name}: {err}", file=sys.stderr)
    return {
        "model": name,
        "url": url,
        "api_style": target["style"],
        "ok": False,
        "latency_s": None,
        "tokens_in": None,
        "tokens_out": None,
        "tokens_source": None,
        "tokens_per_s": None,
        "chars": None,
        "lines": None,
        "backend": None,
        "finish_reason": None,
        "reasoning_chars": None,
        "used_reasoning_as_answer": None,
        "response": None,
        "error": err,
    }


def print_summary(results: list) -> None:
    header = (
        f"{'MODELO/SERVICIO':<24} {'API':<6} {'OK':<4} {'LAT (s)':>9} {'TOK OUT':>8} "
        f"{'TOK/S':>8} {'FINISH':<8} {'RAZON?':<7}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<24} "
            f"{(r.get('api_style') or '-'):<6} "
            f"{('si' if r['ok'] else 'NO'):<4} "
            f"{(r['latency_s'] if r['latency_s'] is not None else '-'):>9} "
            f"{(r['tokens_out'] if r['tokens_out'] is not None else '-'):>8} "
            f"{(r['tokens_per_s'] if r['tokens_per_s'] is not None else '-'):>8} "
            f"{(r.get('finish_reason') or '-'):<8} "
            f"{('si' if r.get('used_reasoning_as_answer') else '-'):<7}"
        )
    print(
        "\nFINISH: 'length' = la respuesta se corto por --max-tokens | "
        "'stop' = termino natural.\nRAZON?: 'si' = la API devolvio vacio en 'content' "
        "y el texto procede de 'reasoning' (modelo de razonamiento)."
    )


def infer_style(url: str) -> str:
    """Estilo por defecto a partir de la URL si el usuario no lo indica."""
    low = url.lower().rstrip("/")
    if low.endswith("/v1/chat/completions") or low.endswith("/v1/completions"):
        return "openai"
    if low.endswith("/api/chat"):
        return "dual"
    if low.endswith("/chat"):
        return "body"
    return "openai"


def parse_targets(args, base_url: str) -> list:
    """Construye la lista de destinos (maximo 2) a partir de --service o de --model (modo legado)."""
    targets = []
    if args.service:
        for spec in args.service:
            if "=" not in spec:
                raise ValueError(f"--service '{spec}' debe ser NOMBRE=URL")
            name, url = spec.split("=", 1)
            url = url.strip()
            if "@" in url and url.rsplit("@", 1)[-1].split("://")[0] in API_STYLES:
                url, style = url.rsplit("@", 1)
            else:
                style = args.api or infer_style(url)
            if not name.strip() or not url.startswith("http"):
                raise ValueError(f"--service '{spec}' no es valido (NOMBRE=URL)")
            # En estilo openai el nombre del destino se reutiliza como "model" del payload
            # (necesario en endpoints OpenAI-compatibles como los de Ollama: 11434/11435).
            targets.append(
                {"name": name.strip(), "url": url.rstrip("/"), "style": style,
                 "model": name.strip() if style == "openai" else None}
            )
        return targets

    # Modo legado: --model contra AZURE_VM_URL + --path (API compatible con OpenAI)
    style = args.api or "openai"
    url = base_url.rstrip("/") + args.path
    for model in args.model or []:
        targets.append({"name": model, "url": url, "style": style, "model": model})
    return targets


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ejecuta el mismo prompt en 1-2 modelos/servicios de la VM y guarda respuestas + metricas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        nargs="+",
        metavar="NOMBRE",
        help="modo legado OpenAI: uno o dos nombres de modelo contra AZURE_VM_URL + --path",
    )
    p.add_argument(
        "--service",
        action="append",
        metavar="NOMBRE=URL[@ESTILO]",
        help="destino explicito de la VM, repetible hasta 2 veces. "
        "Ej: --service dual=http://127.0.0.1:18001/api/chat  --service a=query@http://127.0.0.1:8000/chat",
    )
    p.add_argument(
        "--api",
        choices=API_STYLES,
        default=None,
        help="estilo de API para todos los destinos (si se omite se infiere de la URL; legado=openai)",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="prompt inline")
    p.add_argument("--prompt-file", help="lee el prompt de un archivo (tiene prioridad sobre --prompt)")
    p.add_argument("--system", default=None, help="mensaje de sistema opcional (solo estilo openai; en los demas se antepone al prompt)")
    p.add_argument("--max-tokens", type=int, default=512, help="maximo de tokens de salida")
    p.add_argument("--temperature", type=float, default=0.2, help="temperatura de muestreo")
    p.add_argument("--timeout", type=float, default=120.0, help="timeout HTTP en segundos")
    p.add_argument("--path", default=DEFAULT_PATH, help="ruta del endpoint relativa a AZURE_VM_URL (modo legado)")
    p.add_argument("--out", default=None, help="ruta del JSON de salida (por defecto results/results_<ts>.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.model and not args.service:
        print(
            "ERROR: falta indicar contra que ejecutar el prompt.\n"
            "  Modo legado (OpenAI):  --model NOMBRE [NOMBRE]   (+ AZURE_VM_URL y --path)\n"
            "  Modo VM:               --service NOMBRE=URL      (repetible, maximo 2)",
            file=sys.stderr,
        )
        return 2

    base_url = os.environ.get("AZURE_VM_URL", "").strip()
    if args.model and not base_url:
        print(
            "ERROR: falta la variable de entorno AZURE_VM_URL (necesaria con --model).\n"
            '  PowerShell: $env:AZURE_VM_URL="http://127.0.0.1:11434"\n'
            "  bash:       export AZURE_VM_URL=\"http://127.0.0.1:11434\"",
            file=sys.stderr,
        )
        return 2

    try:
        targets = parse_targets(args, base_url)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not targets:
        print("ERROR: no hay destinos que ejecutar.", file=sys.stderr)
        return 2
    if len(targets) > 2:
        print("ERROR: se admiten como maximo 2 destinos (comparacion A vs B).", file=sys.stderr)
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

    print(f"Estilo   : {', '.join(sorted({t['style'] for t in targets}))}")
    print("Destinos :")
    for t in targets:
        print(f"  - {t['name']:<20} [{t['style']}] {t['url']}")
    print(f"Prompt   : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")

    results = []
    for i, target in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] ejecutando '{target['name']}' ...")
        results.append(run_model_safe(target, messages, args))

    print_summary(results)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or os.path.join("results", f"results_{stamp}.json")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "targets": [{"name": t["name"], "url": t["url"], "api_style": t["style"]} for t in targets],
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
