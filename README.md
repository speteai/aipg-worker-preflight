# aipg-worker-preflight

Find the **safe advertised context** for an [AI Power Grid](https://github.com/AIPowerGrid/grid-text-worker)
text worker — before the grid sends you a job your GPU can't start in time.

## The problem it solves

grid-text-worker streams every generation through one httpx client with a fixed
**120 s read timeout** (`ws_client.py:130`). During prompt **prefill**, no bytes
arrive before the first token — so if your backend needs longer than 120 s to
prefill a prompt, the client raises `ReadTimeout`. The worker logs it as
*"Backend unreachable"*, marks a **healthy** backend offline, and drops **off the
grid** (see [grid-text-worker#47](https://github.com/AIPowerGrid/grid-text-worker/issues/47)).

At the same time the worker advertises the model's **native** context (e.g.
`40960` from Ollama `/api/show`) without checking whether your GPU can actually
prefill that much in time. Prefill cost grows ~quadratically with prompt length,
so on a mid-range GPU a ~22 k-token prompt can take **minutes** — and every such
job fails, repeatedly, on the same worker.

The manual fix is to cap `max_context` in the worker settings — but operators are
left **guessing** the number. This tool **measures** it.

## What it does

1. Detects your backend (Ollama or any OpenAI-compatible server) and its native
   context.
2. Sends a ladder of increasing prompts (`max_tokens=1`) and measures real
   prefill time — using Ollama's exact `prompt_eval_duration` where available,
   wall-clock otherwise — against the actual prompt-token count the backend
   reports.
3. Prints the largest context that prefills within a safety fraction (default
   75 %) of the 120 s timeout, as the `max_context` value to set.

It talks **only to your backend**. No grid API key, no calls to the grid, no
dependencies — stdlib Python 3.8+.

## Usage

```bash
# Ollama (default URL http://localhost:11434)
python3 preflight.py --model qwen3:8b

# vLLM / LM Studio / any OpenAI-compatible server
python3 preflight.py --url http://localhost:8000/v1 --model my-model --engine openai

# design against a different timeout / safety margin
python3 preflight.py --model qwen3:8b --timeout 120 --safety 0.75
```

### Example output

```
Backend : http://localhost:11434  (engine: ollama)
Model   : qwen3:8b
Native context reported by backend: 40960
Prefill budget: 90s  (75% of 120s read timeout)

  target  prompt_tok  prefill_s   verdict
------------------------------------------------
     512         511        5.4   OK
    2048        1931       22.1   OK
    4096        3831       48.7   OK
    8192        7704       96.3   OVER budget
================================================
RECOMMENDED max_context = 7168
================================================
Set it in the worker so the grid won't send jobs you can't start in time:
  • Web/Settings:   max_context = 7168
  • Headless/env:   GRID_MAX_CONTEXT_LENGTH=7168
```

(Numbers are illustrative; yours depend on your GPU, model and quant.)

## Notes

- If your backend prefills its full native context within budget, the tool tells
  you so and recommends leaving `max_context` at `0` (auto).
- Reasoning models return the prefill cost the same way; this measures prefill,
  independent of the reasoning-output behaviour.
- The real fix belongs in the worker (a time-to-first-token timeout and a
  startup check), as proposed in grid-text-worker#47. Until then, this gives
  operators the right number instead of a guess.

## License

MIT — see [LICENSE](LICENSE). Unofficial, community-built; not affiliated with
AI Power Grid.
