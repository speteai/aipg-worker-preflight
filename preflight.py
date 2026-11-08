#!/usr/bin/env python3
"""
aipg-worker-preflight — find the safe advertised context for a grid-text-worker backend.

Background
----------
grid-text-worker streams generations through a single httpx client with a fixed
120 s read timeout (ws_client.py:130). During prompt *prefill* no bytes arrive
before the first token, so if the backend needs longer than that timeout to
prefill a prompt, the client raises ReadTimeout. The worker reports it as
"Backend unreachable", marks a healthy backend offline, and goes off the grid
(see AIPowerGrid/grid-text-worker#47).

Meanwhile the worker advertises the model's *native* context (e.g. 40960 from
Ollama /api/show) without checking whether the GPU can actually prefill that
much in time. On a mid-range GPU a 22k-token prompt can take ~8 min to prefill
and every such job fails.

This tool benchmarks your backend's real prefill speed and tells you the largest
context that reliably prefills within the timeout — the value to put in the
worker's `max_context` setting (or GRID_MAX_CONTEXT_LENGTH) so the grid never
sends you a job you can't start in time.

It only talks to YOUR backend. No grid API key, no network calls to the grid.
Stdlib only — no pip install.
"""
import argparse, json, sys, time, urllib.request, urllib.error

UA = "aipg-worker-preflight/1.0"


def http(method, url, payload=None, api_key="", timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("User-Agent", UA)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        body = json.loads(r.read().decode())
        return r.status, body, time.time() - t0
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode()), time.time() - t0
        except Exception:
            return e.code, None, time.time() - t0
    except Exception as e:
        return None, {"_err": str(e)}, time.time() - t0


def base_url(url):
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def detect_engine(url, api_key):
    """Guess ollama vs openai-compatible by probing endpoints."""
    b = base_url(url)
    st, _, _ = http("GET", b + "/api/tags", api_key=api_key, timeout=10)
    if st == 200:
        return "ollama"
    st, _, _ = http("GET", b + "/v1/models", api_key=api_key, timeout=10)
    if st == 200:
        return "openai"
    return None


def native_context(url, engine, model, api_key):
    b = base_url(url)
    if engine == "ollama":
        st, d, _ = http("POST", b + "/api/show", {"name": model}, api_key, timeout=15)
        if st == 200 and isinstance(d, dict):
            for k, v in (d.get("model_info") or {}).items():
                if k.endswith(".context_length"):
                    return int(v)
    else:
        st, d, _ = http("GET", b + "/v1/models", api_key=api_key, timeout=15)
        if st == 200 and isinstance(d, dict):
            for m in d.get("data", []):
                if model and m.get("id") != model:
                    continue
                if m.get("max_model_len"):
                    return int(m["max_model_len"])
    return None


def make_prompt(approx_tokens):
    # ~1.3 tokens per "word " chunk for most tokenizers; overshoot slightly,
    # the backend reports the real count which is what we actually plot.
    chunk = "The quick brown fox jumps over the lazy dog. "
    words = max(1, int(approx_tokens / 1.2))
    reps = max(1, int(words / 9) + 1)
    return (chunk * reps)


def probe_once(url, engine, model, api_key, approx_tokens, timeout):
    """Send one max-1-token request. Return (prompt_tokens, prefill_seconds) or None."""
    b = base_url(url)
    prompt = make_prompt(approx_tokens)
    if engine == "ollama":
        st, d, wall = http("POST", b + "/api/generate",
                           {"model": model, "prompt": prompt, "stream": False,
                            "options": {"num_predict": 1}}, api_key, timeout=timeout)
        if st == 200 and isinstance(d, dict):
            toks = d.get("prompt_eval_count")
            # Ollama reports prefill duration directly, in nanoseconds — exact.
            dur = d.get("prompt_eval_duration")
            secs = (dur / 1e9) if dur else wall
            return (int(toks) if toks else None, secs)
        return (None, None) if st else (None, None)
    else:
        st, d, wall = http("POST", b + "/v1/chat/completions",
                           {"model": model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 1, "stream": False, "temperature": 0}, api_key, timeout=timeout)
        if st == 200 and isinstance(d, dict):
            toks = (d.get("usage") or {}).get("prompt_tokens")
            return (int(toks) if toks else None, wall)  # wall ≈ prefill for 1 output token
        return (None, None)


def main():
    ap = argparse.ArgumentParser(description="Find the safe advertised context for a grid-text-worker backend.")
    ap.add_argument("--url", default="http://localhost:11434", help="Backend base URL (default Ollama localhost:11434)")
    ap.add_argument("--model", required=True, help="Model name the backend expects (e.g. qwen3:8b)")
    ap.add_argument("--engine", choices=["auto", "ollama", "openai"], default="auto")
    ap.add_argument("--api-key", default="", help="Backend API key, if it needs one")
    ap.add_argument("--timeout", type=int, default=120, help="Worker read timeout to design against (s, default 120)")
    ap.add_argument("--safety", type=float, default=0.75, help="Fraction of the timeout to target for prefill (default 0.75)")
    args = ap.parse_args()

    engine = args.engine
    if engine == "auto":
        engine = detect_engine(args.url, args.api_key)
        if not engine:
            print(f"ERROR: could not reach a backend at {args.url} (tried Ollama /api/tags and OpenAI /v1/models).")
            sys.exit(2)
    print(f"Backend : {args.url}  (engine: {engine})")
    print(f"Model   : {args.model}")

    native = native_context(args.url, engine, args.model, args.api_key)
    print(f"Native context reported by backend: {native if native else 'unknown'}")

    budget = args.timeout * args.safety
    print(f"Prefill budget: {budget:.0f}s  ({args.safety:.0%} of {args.timeout}s read timeout)\n")

    sizes = [512, 1024, 2048, 4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536]
    if native:
        sizes = [s for s in sizes if s < native] + [native]

    print(f"{'target':>8} {'prompt_tok':>11} {'prefill_s':>10}   verdict")
    print("-" * 48)
    results = []
    for s in sizes:
        toks, secs = probe_once(args.url, engine, args.model, args.api_key, s, args.timeout + 60)
        if toks is None or secs is None:
            print(f"{s:>8} {'—':>11} {'—':>10}   request failed (timeout or error) — stopping")
            break
        ok = secs <= budget
        print(f"{s:>8} {toks:>11} {secs:>10.1f}   {'OK' if ok else 'OVER budget'}")
        results.append((toks, secs, ok))
        if not ok:
            break  # once over budget, larger prompts only get worse

    if not results:
        print("\nNo successful probes — check the backend URL/model.")
        sys.exit(2)

    safe = [t for (t, s, ok) in results if ok]
    if not safe:
        rec = results[0][0]
        print(f"\n⚠  Even {rec} tokens prefills slower than the budget on this backend.")
        print("   This GPU/model is very slow at prefill; consider a smaller model or quant.")
    else:
        largest_safe = max(safe)
        # round down to a tidy multiple of 1024 for a stable advertised value
        rec = (largest_safe // 1024) * 1024 or largest_safe
        if native and rec >= native:
            print(f"\n✓ This backend prefills its full native context ({native}) within budget.")
            print("  No cap needed — you can leave max_context at 0 (auto).")
            return
    print("\n" + "=" * 48)
    print(f"RECOMMENDED max_context = {rec}")
    print("=" * 48)
    print("Set it in the worker so the grid won't send jobs you can't start in time:")
    print(f"  • Web/Settings:   max_context = {rec}")
    print(f"  • Headless/env:   GRID_MAX_CONTEXT_LENGTH={rec}")
    print("\nWhy: prefill time grows ~quadratically with prompt length, and the worker's")
    print(f"120 s read timeout fires during prefill (see grid-text-worker#47). Capping the")
    print("advertised context keeps every accepted job startable within that window.")


if __name__ == "__main__":
    main()
