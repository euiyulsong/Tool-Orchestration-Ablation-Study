#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4-way ablation for UI/tool orchestration with GPT-6 Luna. (BFCL/OpenAI adapter v2)

Goal
----
Reproduce patterns like:

    user-facing generation
      -> tool call / widget
      -> next action / conclusion / recommendation

and compare WHERE the post-tool "next action" is generated.

Strategies
----------
A. POST_TOOL_CONTINUATION
   LLM #1 -> tool call only
   execute tool
   LLM #2 -> natural continuation grounded in tool result

B. PRETEXT_TOOL_CONTINUATION
   LLM #1 -> may generate short user-facing pre-text + tool call
   execute tool
   LLM #2 -> natural continuation grounded in tool result

C. TOOL_WITH_NEXT_ACTION
   LLM #1 -> may generate pre-text + tool call whose arguments ALSO contain
             `next_action`
   execute tool
   NO second LLM call
   render: pre-text -> widget/tool result -> planned next_action

D. COMBINED_ONE_PASS
   LLM #1 -> tool call whose arguments contain BOTH `pre_text` and `next_action`
   execute tool
   NO second LLM call
   render: pre_text -> widget/tool result -> next_action

Important experimental property
-------------------------------
The continuation prompt is intentionally NOT stuffed with strings such as
"하나만 고르면". A/B simply reuse a small, general base policy after the tool
result. C/D are allowed to choose their own natural wording.

Dataset
-------
BFCL v4 (Berkeley Function Calling Leaderboard), Apache-2.0.
We sample 50 diverse positive tool-use examples from:
  - simple_python
  - multiple
  - parallel

The BFCL function schemas are used as the actual OpenAI tools.

Tool execution
--------------
BFCL contains function-call tasks but not a production backend for every API.
To isolate orchestration quality from external API/network failures, this script
uses a deterministic MOCK executor for every selected tool call.

The same mock result is derived from the same function name + arguments across
all strategies. This is deliberate: we want to compare orchestration, not API
availability.

Outputs
-------
ablation_4way_outputs/
  samples.json
  results.jsonl
  results.csv
  summary.csv
  png/
    page_01.png ... page_10.png

Each PNG contains 5 samples and shows A/B/C/D side by side.

Install
-------
python -m pip install -U openai pillow

Run
---
export OPENAI_API_KEY="..."
python tool_orchestration_ablation_4way.py

Optional:
python tool_orchestration_ablation_4way.py --n 50 --seed 42
python tool_orchestration_ablation_4way.py --dataset-dir ./bfcl_cache
python tool_orchestration_ablation_4way.py --judge
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import textwrap
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


MODEL = "gpt-6-luna"

BFCL_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

BFCL_FILES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
}

BASE_POLICY = """
You are a helpful assistant using application tools.

Use the provided tool when it is needed to answer the user's request.
Do not invent tool results.

When a tool result is available, respond naturally based on that result.
Keep the response concise and useful for the user's actual request.
Do not force a fixed phrase, fixed conclusion format, or fixed recommendation format.
""".strip()

PROMPT_A_FIRST = BASE_POLICY + """

For this step, call the needed tool directly.
Do not add user-facing text before the tool call.
""".strip()

PROMPT_B_FIRST = BASE_POLICY + """

If natural, you may write a brief user-facing sentence before calling the tool.
Then call the needed tool.
""".strip()

PROMPT_C_FIRST = BASE_POLICY + """

Call the needed tool.
The tool schema contains a `next_action` field added by the application.
Fill `next_action` with the natural user-facing text you would show AFTER the
tool/widget is rendered.

Because the tool has not run yet, do not claim specific unseen result values.
The next action may be a short conclusion, recommendation, explanation, or
follow-up, depending on the user's request. Use natural wording.
""".strip()

PROMPT_D_FIRST = BASE_POLICY + """

Complete the user-facing tool interaction in one model generation.
The tool schema contains two application-only fields:
- `pre_text`: optional short text shown before the tool/widget
- `next_action`: text shown after the tool/widget

Fill them naturally for this request.
Do not use canned phrases.
Because the tool has not run yet, do not claim specific unseen result values.
""".strip()

JUDGE_POLICY = """
You compare four UI/tool orchestration outputs for the same user request.

The tool/widget result shown to the user is authoritative.
Score each strategy from 1 to 5 for:
- task_helpfulness
- natural_flow
- grounding_to_tool_result
- unnecessary_repetition (5 means no unnecessary repetition)

Do not reward verbosity by itself.
Return strict JSON only with keys A, B, C, D, each containing the four scores,
plus `best` ("A"|"B"|"C"|"D"|"tie") and a short `reason`.
""".strip()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now() -> float:
    return time.perf_counter()


def stable_int(obj: Any) -> int:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:12], 16)


def safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def obj_dump(x: Any) -> Any:
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if isinstance(x, dict):
        return x
    return str(x)


def get_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    d = obj_dump(usage)
    if not isinstance(d, dict):
        return {"raw": str(d)}
    return d


def usage_total_tokens(usage: Dict[str, Any]) -> int:
    for k in ("total_tokens", "total"):
        if isinstance(usage.get(k), int):
            return usage[k]
    inp = usage.get("input_tokens", 0) or 0
    out = usage.get("output_tokens", 0) or 0
    return int(inp) + int(out)


# ---------------------------------------------------------------------------
# BFCL loading
# ---------------------------------------------------------------------------

def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"[download] {url}")
    urllib.request.urlretrieve(url, path)


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_question(q: Any) -> str:
    """
    BFCL v4 question can be nested chat-like structures.
    Extract the user-visible query conservatively.
    """
    if isinstance(q, str):
        return q

    texts: List[str] = []

    def rec(x: Any):
        if isinstance(x, str):
            texts.append(x)
        elif isinstance(x, list):
            for y in x:
                rec(y)
        elif isinstance(x, dict):
            role = x.get("role") or x.get("from")
            if role in (None, "user", "human"):
                if "content" in x:
                    rec(x["content"])
                elif "value" in x:
                    rec(x["value"])
            elif "content" in x and isinstance(x["content"], str):
                # Some BFCL structures do not mark role consistently.
                texts.append(x["content"])

    rec(q)
    # Prefer the longest human-like string if nesting duplicated content.
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        return safe_json(q)
    return max(texts, key=len)


OPENAI_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")

BFCL_TYPE_MAP = {
    "dict": "object",
    "object": "object",
    "list": "array",
    "tuple": "array",
    "array": "array",
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "null": "null",
    "none": "null",
}


def sanitize_tool_name(raw_name: Any) -> str:
    """
    OpenAI function names must match ^[a-zA-Z0-9_-]+$.
    BFCL intentionally contains Python-style names such as:
      math.factorial
      vegan_restaurant.find_nearby

    Convert punctuation / namespace separators to "__".
    Keep it deterministic because the name is part of the experiment.
    """
    raw = str(raw_name or "tool").strip()
    safe = OPENAI_NAME_RE.sub("__", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "tool"

    # Keep room for a collision suffix if needed.
    return safe[:60]


def _sanitize_schema_node(node: Any) -> Any:
    """
    Convert BFCL's Python-ish type schema into valid JSON Schema.

    Examples:
      {"type": "dict"}  -> {"type": "object"}
      {"type": "float"} -> {"type": "number"}

    BFCL also uses nested array/items schemas, so sanitize recursively.
    Unknown Python-ish type names are removed rather than forwarded as an
    invalid JSON-Schema `type`.
    """
    if isinstance(node, list):
        return [_sanitize_schema_node(x) for x in node]

    if not isinstance(node, dict):
        return copy.deepcopy(node)

    out = {}

    for k, v in node.items():
        if k == "type":
            if isinstance(v, str):
                mapped = BFCL_TYPE_MAP.get(v.lower())
                if mapped is not None:
                    out["type"] = mapped
                # Unknown type: omit it. Description/enum still constrain it.
            elif isinstance(v, list):
                mapped_types = []
                for t in v:
                    if isinstance(t, str):
                        mt = BFCL_TYPE_MAP.get(t.lower())
                        if mt is not None and mt not in mapped_types:
                            mapped_types.append(mt)
                if mapped_types:
                    out["type"] = mapped_types
            continue

        if k == "properties":
            if isinstance(v, dict):
                out["properties"] = {
                    str(pk): _sanitize_schema_node(pv)
                    for pk, pv in v.items()
                }
            else:
                out["properties"] = {}
            continue

        if k == "items":
            out["items"] = _sanitize_schema_node(v)
            continue

        # `default`, `enum`, descriptions, bounds, etc. are legal JSON Schema
        # keywords and can be preserved.
        out[k] = _sanitize_schema_node(v)

    # Infer container type from structure if BFCL omitted / used unknown type.
    if "properties" in out and "type" not in out:
        out["type"] = "object"
    if "items" in out and "type" not in out:
        out["type"] = "array"

    if out.get("type") == "object":
        props = out.get("properties")
        if not isinstance(props, dict):
            props = {}
        out["properties"] = props

        required = out.get("required", [])
        if not isinstance(required, list):
            required = []
        # OpenAI rejects required keys that do not exist in properties.
        out["required"] = [
            str(x) for x in required
            if str(x) in props
        ]

        # We use non-strict tool calling for compatibility with BFCL optional
        # parameters, but explicit additionalProperties=False is still useful.
        out.setdefault("additionalProperties", False)

    return out


def sanitize_parameters(params: Any) -> dict:
    """
    Return an OpenAI-compatible JSON Schema object for function parameters.
    """
    if not isinstance(params, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    p = _sanitize_schema_node(copy.deepcopy(params))

    # Function parameters must be a top-level object schema.
    if not isinstance(p, dict):
        p = {}
    p["type"] = "object"
    p.setdefault("properties", {})
    if not isinstance(p["properties"], dict):
        p["properties"] = {}

    required = p.get("required", [])
    if not isinstance(required, list):
        required = []
    p["required"] = [
        str(x) for x in required
        if str(x) in p["properties"]
    ]
    p.setdefault("additionalProperties", False)

    return p


def normalize_tool(fn: dict) -> dict:
    raw_name = fn.get("name") or fn.get("function_name") or "tool"
    name = sanitize_tool_name(raw_name)
    desc = fn.get("description") or f"Call {raw_name}."
    params = sanitize_parameters(fn.get("parameters", {}))

    # Preserve the original BFCL namespace in the description so replacing
    # dots in the callable name does not remove semantic context.
    if str(raw_name) != name:
        desc = f"Original BFCL function: {raw_name}. {desc}"

    return {
        "type": "function",
        "name": name,
        "description": str(desc)[:4000],
        "parameters": params,
        # Explicit non-strict mode is deliberate: BFCL has optional params and
        # schemas that were not authored for OpenAI Structured Outputs.
        "strict": False,
    }


def tool_names(sample: dict) -> List[str]:
    return [str(x.get("name", "")) for x in sample.get("function", [])]


def select_diverse(rows_by_cat: Dict[str, List[dict]], n: int, seed: int) -> List[dict]:
    """
    Stratified + greedy unique-tool selection.
    Default for n=50:
      simple_python 20, multiple 15, parallel 15
    """
    rng = random.Random(seed)
    cats = list(rows_by_cat.keys())

    if n == 50 and set(cats) >= {"simple_python", "multiple", "parallel"}:
        quotas = {"simple_python": 20, "multiple": 15, "parallel": 15}
    else:
        base = n // len(cats)
        quotas = {c: base for c in cats}
        for c in cats[: n - base * len(cats)]:
            quotas[c] += 1

    selected = []
    seen_names = set()

    for cat in cats:
        rows = list(rows_by_cat[cat])
        rng.shuffle(rows)
        quota = quotas.get(cat, 0)

        # First pass: maximize new function names.
        chosen = []
        for row in rows:
            names = set(tool_names(row))
            if names - seen_names:
                chosen.append(row)
                seen_names.update(names)
                if len(chosen) >= quota:
                    break

        # Fill remaining quota.
        if len(chosen) < quota:
            ids = {id(x) for x in chosen}
            for row in rows:
                if id(row) in ids:
                    continue
                chosen.append(row)
                if len(chosen) >= quota:
                    break

        for row in chosen[:quota]:
            out = copy.deepcopy(row)
            out["_category"] = cat
            selected.append(out)

    rng.shuffle(selected)
    return selected[:n]


def load_bfcl_samples(dataset_dir: Path, n: int, seed: int) -> List[dict]:
    rows_by_cat = {}
    for cat, filename in BFCL_FILES.items():
        path = dataset_dir / filename
        download(f"{BFCL_BASE}/{filename}", path)
        rows_by_cat[cat] = load_jsonl(path)

    raw = select_diverse(rows_by_cat, n=n, seed=seed)

    out = []
    for i, row in enumerate(raw, 1):
        tools = [normalize_tool(x) for x in row.get("function", [])]
        if not tools:
            continue

        # Two different BFCL names could collapse to the same sanitized name.
        # Make names unique within the request deterministically.
        seen = {}
        for t in tools:
            base = t["name"]
            count = seen.get(base, 0)
            if count:
                suffix = hashlib.sha1(
                    (base + "|" + t.get("description", "")).encode("utf-8")
                ).hexdigest()[:6]
                t["name"] = f"{base[:53]}_{suffix}"
            seen[base] = count + 1

        out.append({
            "sample_id": i,
            "bfcl_id": row.get("id"),
            "category": row.get("_category"),
            "query": flatten_question(row.get("question")),
            "tools": tools,
        })
    return out[:n]


# ---------------------------------------------------------------------------
# Tool schema variants
# ---------------------------------------------------------------------------

def add_application_fields(
    tools: List[dict],
    add_pre_text: bool = False,
    add_next_action: bool = False,
) -> List[dict]:
    out = copy.deepcopy(tools)

    for t in out:
        p = t["parameters"]
        p.setdefault("type", "object")
        p.setdefault("properties", {})
        props = p["properties"]

        if add_pre_text:
            props["pre_text"] = {
                "type": "string",
                "description": (
                    "Application UI text shown immediately before this tool/widget. "
                    "Keep it short and natural. It is not passed to the backend."
                ),
            }

        if add_next_action:
            props["next_action"] = {
                "type": "string",
                "description": (
                    "Application UI text shown after this tool/widget. "
                    "It is a natural next action, conclusion, recommendation, or "
                    "follow-up. It is not passed to the backend."
                ),
            }

        # Don't force our UI fields into `required`.
        # The prompt requests them; keeping them optional avoids mutating the
        # original BFCL task contract more than necessary.
        p["additionalProperties"] = False
    return out


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(response: Any) -> Tuple[str, List[dict]]:
    """
    Returns (all output_text, function calls).
    """
    text = getattr(response, "output_text", "") or ""
    calls = []

    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) == "function_call":
            args_raw = getattr(item, "arguments", "{}") or "{}"
            try:
                args = json.loads(args_raw)
            except Exception:
                args = {"__raw_arguments__": args_raw}
            calls.append({
                "call_id": getattr(item, "call_id", None),
                "name": getattr(item, "name", None),
                "arguments": args,
            })

    return text.strip(), calls


def strip_ui_fields(args: dict) -> Tuple[dict, str, str]:
    args = copy.deepcopy(args)
    pre_text = args.pop("pre_text", "") if isinstance(args, dict) else ""
    next_action = args.pop("next_action", "") if isinstance(args, dict) else ""
    return args, str(pre_text or "").strip(), str(next_action or "").strip()


# ---------------------------------------------------------------------------
# Deterministic mock tool executor
# ---------------------------------------------------------------------------

def mock_execute(name: str, args: dict) -> dict:
    """
    Deterministic semantic mock output.
    It NEVER performs external side effects.

    The result shape changes by tool/function semantics to make the 50 examples
    more realistic for human inspection.
    """
    key = f"{name} {safe_json(args)}".lower()
    h = stable_int({"name": name, "args": args})
    rng = random.Random(h)

    base = {
        "ok": True,
        "tool": name,
        "request": args,
        "mock": True,
    }

    # Search / retrieval / list
    if any(w in key for w in ["search", "find", "lookup", "query", "list", "retrieve"]):
        base["results"] = [
            {
                "id": f"r{i+1}",
                "label": f"Result {i+1}",
                "score": round(0.91 - i * 0.07 + rng.uniform(-0.015, 0.015), 3),
                "summary": f"Mock result {i+1} relevant to {name}",
            }
            for i in range(3)
        ]
        return base

    # Weather
    if any(w in key for w in ["weather", "forecast", "temperature", "climate"]):
        base["result"] = {
            "condition": rng.choice(["Clear", "Cloudy", "Light rain"]),
            "temperature_c": rng.randint(8, 29),
            "humidity_pct": rng.randint(35, 82),
        }
        return base

    # Maps / place / location
    if any(w in key for w in ["map", "place", "location", "restaurant", "hotel", "nearby", "geo"]):
        lat = 37.5 + rng.random() * 0.12
        lon = 126.9 + rng.random() * 0.12
        base["results"] = [
            {
                "id": f"p{i+1}",
                "name": f"Place {i+1}",
                "rating": round(4.1 + 0.2 * rng.random(), 1),
                "lat": round(lat + i * 0.002, 6),
                "lon": round(lon + i * 0.002, 6),
            }
            for i in range(3)
        ]
        return base

    # Calendar / booking / reservation
    if any(w in key for w in ["calendar", "schedule", "booking", "reservation", "appointment", "availability"]):
        base["result"] = {
            "available": True,
            "slots": ["10:00", "13:30", "16:00"],
            "confirmation_id": f"mock-{h % 100000:05d}",
        }
        return base

    # Finance / market / price
    if any(w in key for w in ["stock", "market", "price", "quote", "currency", "exchange", "finance"]):
        base["result"] = {
            "value": round(50 + (h % 10000) / 137.0, 2),
            "change_pct": round(rng.uniform(-3.0, 3.0), 2),
            "currency": "USD",
        }
        return base

    # Email / message / notification
    if any(w in key for w in ["email", "mail", "message", "notify", "send", "sms"]):
        base["result"] = {
            "status": "accepted",
            "message_id": f"mock-msg-{h % 1000000:06d}",
        }
        return base

    # Create / update / delete side effects
    if any(w in key for w in ["create", "update", "delete", "remove", "insert", "write", "set_"]):
        base["result"] = {
            "status": "simulated_success",
            "object_id": f"mock-obj-{h % 100000:05d}",
        }
        return base

    # Numeric / calculator-like
    if any(w in key for w in ["calculate", "compute", "sum", "convert", "math"]):
        nums = []
        for v in args.values() if isinstance(args, dict) else []:
            if isinstance(v, (int, float)):
                nums.append(float(v))
        base["result"] = {
            "numeric_result": round(sum(nums), 6) if nums else round((h % 100000) / 97.0, 6)
        }
        return base

    # Default
    base["result"] = {
        "status": "success",
        "value": f"mock_value_{h % 10000:04d}",
        "details": f"Deterministic mock execution for {name}",
    }
    return base


def execute_calls(calls: List[dict]) -> List[dict]:
    observations = []
    for call in calls:
        backend_args, pre, nxt = strip_ui_fields(call["arguments"])
        observations.append({
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": backend_args,
            "pre_text": pre,
            "next_action": nxt,
            "result": mock_execute(call["name"], backend_args),
        })
    return observations


# ---------------------------------------------------------------------------
# OpenAI tool-schema preflight
# ---------------------------------------------------------------------------

VALID_JSON_TYPES = {
    "object", "array", "string", "number",
    "integer", "boolean", "null",
}


def validate_schema_node(node: Any, path: str = "$") -> List[str]:
    errors = []
    if not isinstance(node, dict):
        return errors

    typ = node.get("type")
    if isinstance(typ, str) and typ not in VALID_JSON_TYPES:
        errors.append(f"{path}.type={typ!r}")
    elif isinstance(typ, list):
        for t in typ:
            if t not in VALID_JSON_TYPES:
                errors.append(f"{path}.type contains {t!r}")

    if node.get("type") == "object":
        props = node.get("properties", {})
        if not isinstance(props, dict):
            errors.append(f"{path}.properties is not object")
        else:
            for k, v in props.items():
                errors.extend(validate_schema_node(v, f"{path}.properties.{k}"))

        req = node.get("required", [])
        if isinstance(req, list) and isinstance(props, dict):
            missing = [x for x in req if x not in props]
            if missing:
                errors.append(f"{path}.required missing properties: {missing}")

    if "items" in node:
        errors.extend(validate_schema_node(node["items"], f"{path}.items"))

    return errors


def validate_openai_tools(tools: List[dict]) -> None:
    for i, t in enumerate(tools):
        name = t.get("name", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ValueError(f"Invalid OpenAI tool name at {i}: {name!r}")

        errs = validate_schema_node(t.get("parameters", {}), f"tools[{i}].parameters")
        if errs:
            raise ValueError(
                f"Invalid schema for tool {name}: " + "; ".join(errs[:10])
            )


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def call_model(
    client: OpenAI,
    *,
    instructions: str,
    query: str | None = None,
    tools: List[dict] | None = None,
    tool_choice: Any = None,
    previous_response_id: str | None = None,
    input_items: List[dict] | None = None,
) -> Tuple[Any, float]:
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "reasoning": {"effort": "none"},
        "instructions": instructions,
    }

    if query is not None:
        kwargs["input"] = [{"role": "user", "content": query}]
    elif input_items is not None:
        kwargs["input"] = input_items
    else:
        raise ValueError("query or input_items required")

    if tools is not None:
        validate_openai_tools(tools)
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id

    t0 = now()
    response = client.responses.create(**kwargs)
    latency = now() - t0
    return response, latency


def make_function_outputs(observations: List[dict]) -> List[dict]:
    return [
        {
            "type": "function_call_output",
            "call_id": o["call_id"],
            "output": json.dumps(o["result"], ensure_ascii=False),
        }
        for o in observations
        if o.get("call_id")
    ]


# ---------------------------------------------------------------------------
# Strategy A
# ---------------------------------------------------------------------------

def run_A(client: OpenAI, sample: dict) -> dict:
    t0 = now()

    r1, l1 = call_model(
        client,
        instructions=PROMPT_A_FIRST,
        query=sample["query"],
        tools=sample["tools"],
        tool_choice="required",
    )
    pre_text, calls = parse_response(r1)
    obs = execute_calls(calls)

    if not obs:
        return failure_result("A", "No tool call", t0, r1=r1)

    outputs = make_function_outputs(obs)

    r2, l2 = call_model(
        client,
        instructions=BASE_POLICY,
        previous_response_id=r1.id,
        input_items=outputs,
        tools=sample["tools"],
        tool_choice="none",
    )
    final_text, calls2 = parse_response(r2)

    return {
        "strategy": "A",
        "name": "post_tool_continuation",
        "ok": True,
        "pre_text": pre_text,
        "calls": calls,
        "observations": obs,
        "next_action": final_text,
        "extra_calls_after_tool": calls2,
        "latency_first_s": l1,
        "latency_second_s": l2,
        "latency_total_s": now() - t0,
        "api_calls": 2,
        "usage_first": get_usage(r1),
        "usage_second": get_usage(r2),
    }


# ---------------------------------------------------------------------------
# Strategy B
# ---------------------------------------------------------------------------

def run_B(client: OpenAI, sample: dict) -> dict:
    t0 = now()

    r1, l1 = call_model(
        client,
        instructions=PROMPT_B_FIRST,
        query=sample["query"],
        tools=sample["tools"],
        tool_choice="required",
    )
    pre_text, calls = parse_response(r1)
    obs = execute_calls(calls)

    if not obs:
        return failure_result("B", "No tool call", t0, r1=r1)

    outputs = make_function_outputs(obs)

    # Same general policy, no canned continuation phrase.
    r2, l2 = call_model(
        client,
        instructions=BASE_POLICY,
        previous_response_id=r1.id,
        input_items=outputs,
        tools=sample["tools"],
        tool_choice="none",
    )
    final_text, calls2 = parse_response(r2)

    return {
        "strategy": "B",
        "name": "pretext_tool_continuation",
        "ok": True,
        "pre_text": pre_text,
        "calls": calls,
        "observations": obs,
        "next_action": final_text,
        "extra_calls_after_tool": calls2,
        "latency_first_s": l1,
        "latency_second_s": l2,
        "latency_total_s": now() - t0,
        "api_calls": 2,
        "usage_first": get_usage(r1),
        "usage_second": get_usage(r2),
    }


# ---------------------------------------------------------------------------
# Strategy C
# ---------------------------------------------------------------------------

def run_C(client: OpenAI, sample: dict) -> dict:
    t0 = now()
    tools = add_application_fields(
        sample["tools"],
        add_pre_text=False,
        add_next_action=True,
    )

    r1, l1 = call_model(
        client,
        instructions=PROMPT_C_FIRST,
        query=sample["query"],
        tools=tools,
        tool_choice="required",
    )
    pre_text, calls = parse_response(r1)
    obs = execute_calls(calls)

    if not obs:
        return failure_result("C", "No tool call", t0, r1=r1)

    # If multiple calls exist, concatenate model-planned next actions.
    planned = [o["next_action"] for o in obs if o["next_action"]]
    next_action = "\n".join(dict.fromkeys(planned)).strip()

    return {
        "strategy": "C",
        "name": "tool_with_next_action",
        "ok": True,
        "pre_text": pre_text,
        "calls": calls,
        "observations": obs,
        "next_action": next_action,
        "extra_calls_after_tool": [],
        "latency_first_s": l1,
        "latency_second_s": 0.0,
        "latency_total_s": now() - t0,
        "api_calls": 1,
        "usage_first": get_usage(r1),
        "usage_second": {},
    }


# ---------------------------------------------------------------------------
# Strategy D
# ---------------------------------------------------------------------------

def run_D(client: OpenAI, sample: dict) -> dict:
    t0 = now()
    tools = add_application_fields(
        sample["tools"],
        add_pre_text=True,
        add_next_action=True,
    )

    r1, l1 = call_model(
        client,
        instructions=PROMPT_D_FIRST,
        query=sample["query"],
        tools=tools,
        tool_choice="required",
    )
    model_text, calls = parse_response(r1)
    obs = execute_calls(calls)

    if not obs:
        return failure_result("D", "No tool call", t0, r1=r1)

    pre_candidates = [o["pre_text"] for o in obs if o["pre_text"]]
    next_candidates = [o["next_action"] for o in obs if o["next_action"]]

    pre_text = "\n".join(dict.fromkeys(pre_candidates)).strip()
    if model_text:
        pre_text = (model_text + ("\n" if pre_text else "") + pre_text).strip()

    next_action = "\n".join(dict.fromkeys(next_candidates)).strip()

    return {
        "strategy": "D",
        "name": "combined_one_pass",
        "ok": True,
        "pre_text": pre_text,
        "calls": calls,
        "observations": obs,
        "next_action": next_action,
        "extra_calls_after_tool": [],
        "latency_first_s": l1,
        "latency_second_s": 0.0,
        "latency_total_s": now() - t0,
        "api_calls": 1,
        "usage_first": get_usage(r1),
        "usage_second": {},
    }


def failure_result(strategy: str, error: str, t0: float, r1: Any = None) -> dict:
    return {
        "strategy": strategy,
        "name": "failed",
        "ok": False,
        "error": error,
        "pre_text": "",
        "calls": [],
        "observations": [],
        "next_action": "",
        "latency_first_s": 0.0,
        "latency_second_s": 0.0,
        "latency_total_s": now() - t0,
        "api_calls": 1 if r1 is not None else 0,
        "usage_first": get_usage(r1) if r1 is not None else {},
        "usage_second": {},
    }


# ---------------------------------------------------------------------------
# Optional LLM judge
# ---------------------------------------------------------------------------

def render_observation_for_prompt(result: dict) -> str:
    pieces = []
    for o in result.get("observations", []):
        pieces.append(
            f"TOOL={o['name']}\n"
            f"ARGS={safe_json(o['arguments'])}\n"
            f"RESULT={safe_json(o['result'])}"
        )
    return "\n\n".join(pieces)


def visible_render(result: dict) -> str:
    if not result.get("ok"):
        return f"[ERROR] {result.get('error', 'unknown')}"
    obs = render_observation_for_prompt(result)
    return (
        f"PRE_TEXT:\n{result.get('pre_text','')}\n\n"
        f"WIDGET/TOOL:\n{obs}\n\n"
        f"NEXT_ACTION:\n{result.get('next_action','')}"
    )


def judge_four(client: OpenAI, sample: dict, results: Dict[str, dict]) -> dict:
    payload = {
        "query": sample["query"],
        "A": visible_render(results["A"]),
        "B": visible_render(results["B"]),
        "C": visible_render(results["C"]),
        "D": visible_render(results["D"]),
    }
    r, _ = call_model(
        client,
        instructions=JUDGE_POLICY,
        query=json.dumps(payload, ensure_ascii=False),
        tools=None,
        tool_choice=None,
    )
    text = getattr(r, "output_text", "") or ""
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


# ---------------------------------------------------------------------------
# Metrics / serialization
# ---------------------------------------------------------------------------

def total_tokens_for_result(r: dict) -> int:
    return usage_total_tokens(r.get("usage_first", {})) + usage_total_tokens(r.get("usage_second", {}))


def flatten_row(record: dict) -> dict:
    out = {
        "sample_id": record["sample"]["sample_id"],
        "bfcl_id": record["sample"]["bfcl_id"],
        "category": record["sample"]["category"],
        "query": record["sample"]["query"],
        "tool_count": len(record["sample"]["tools"]),
    }

    for s in "ABCD":
        r = record["results"][s]
        out[f"{s}_ok"] = r.get("ok", False)
        out[f"{s}_api_calls"] = r.get("api_calls", 0)
        out[f"{s}_latency_s"] = round(r.get("latency_total_s", 0.0), 4)
        out[f"{s}_tokens"] = total_tokens_for_result(r)
        out[f"{s}_pre_text"] = r.get("pre_text", "")
        out[f"{s}_next_action"] = r.get("next_action", "")
        out[f"{s}_tool_calls"] = safe_json(r.get("calls", []))
        out[f"{s}_tool_results"] = safe_json(r.get("observations", []))

    if "judge" in record:
        out["judge"] = safe_json(record["judge"])
    return out


def summarize(records: List[dict]) -> List[dict]:
    rows = []
    for s in "ABCD":
        vals = [rec["results"][s] for rec in records]
        ok = [r for r in vals if r.get("ok")]
        rows.append({
            "strategy": s,
            "name": ok[0]["name"] if ok else "",
            "n": len(vals),
            "success": len(ok),
            "success_rate": round(len(ok) / max(1, len(vals)), 4),
            "mean_api_calls": round(sum(r.get("api_calls", 0) for r in vals) / max(1, len(vals)), 3),
            "mean_latency_s": round(sum(r.get("latency_total_s", 0.0) for r in vals) / max(1, len(vals)), 4),
            "mean_tokens": round(sum(total_tokens_for_result(r) for r in vals) / max(1, len(vals)), 2),
            "with_pre_text": sum(bool((r.get("pre_text") or "").strip()) for r in vals),
            "with_next_action": sum(bool((r.get("next_action") or "").strip()) for r in vals),
        })
    return rows


# ---------------------------------------------------------------------------
# PNG rendering with Pillow (no matplotlib / numpy dependency)
# ---------------------------------------------------------------------------

def find_font(size: int):
    if ImageFont is None:
        return None

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_px(text: str, width_chars: int = 55) -> str:
    if not text:
        return ""
    out = []
    for para in str(text).splitlines():
        if not para:
            out.append("")
        else:
            out.extend(
                textwrap.wrap(
                    para,
                    width=width_chars,
                    break_long_words=False,
                    break_on_hyphens=False,
                ) or [""]
            )
    return "\n".join(out)


def compact_tool_result(result: dict, max_chars: int = 700) -> str:
    obs = result.get("observations", [])
    blocks = []
    for o in obs[:3]:
        blocks.append(
            f"{o['name']}({safe_json(o['arguments'])})\n"
            f"=> {safe_json(o['result'])}"
        )
    text = "\n".join(blocks)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def render_png_pages(records: List[dict], out_dir: Path, per_page: int = 5):
    if Image is None:
        print("[png] Pillow unavailable; skipping PNG generation")
        return

    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    title_font = find_font(24)
    body_font = find_font(16)
    small_font = find_font(14)

    W = 2200
    margin = 35
    col_gap = 20
    cols = 4
    col_w = (W - 2 * margin - (cols - 1) * col_gap) // cols

    row_h = 720
    header_h = 160

    pages = math.ceil(len(records) / per_page)

    for page_idx in range(pages):
        chunk = records[page_idx * per_page:(page_idx + 1) * per_page]
        H = header_h + row_h * len(chunk) + margin

        img = Image.new("RGB", (W, H), "white")
        draw = ImageDraw.Draw(img)

        draw.text(
            (margin, 25),
            f"GPT-6 Luna Tool Orchestration 4-Way Ablation  |  Page {page_idx+1}/{pages}",
            fill="black",
            font=title_font,
        )
        draw.text(
            (margin, 65),
            "A: tool -> continuation   B: pre-text + tool -> continuation   "
            "C: tool(next_action)   D: combined pre_text + tool + next_action",
            fill="black",
            font=small_font,
        )

        for ri, rec in enumerate(chunk):
            y0 = header_h + ri * row_h
            sample = rec["sample"]
            q = f'#{sample["sample_id"]:02d} [{sample["category"]}] {sample["query"]}'
            draw.text((margin, y0), wrap_px(q, 120), fill="black", font=body_font)

            y_box = y0 + 65
            labels = [
                ("A", rec["results"]["A"]),
                ("B", rec["results"]["B"]),
                ("C", rec["results"]["C"]),
                ("D", rec["results"]["D"]),
            ]

            for ci, (label, r) in enumerate(labels):
                x = margin + ci * (col_w + col_gap)
                draw.rectangle(
                    [x, y_box, x + col_w, y_box + row_h - 90],
                    outline="black",
                    width=1,
                )

                header = (
                    f"{label} | {r.get('name','')}\n"
                    f"ok={r.get('ok')}  api={r.get('api_calls',0)}  "
                    f"lat={r.get('latency_total_s',0):.2f}s  "
                    f"tok={total_tokens_for_result(r)}"
                )
                draw.multiline_text(
                    (x + 10, y_box + 10),
                    header,
                    fill="black",
                    font=small_font,
                    spacing=4,
                )

                body = (
                    f"PRE\n{r.get('pre_text','')}\n\n"
                    f"TOOL\n{compact_tool_result(r, 500)}\n\n"
                    f"NEXT\n{r.get('next_action','')}"
                )
                body = wrap_px(body, 52)
                if len(body) > 2200:
                    body = body[:2197] + "..."
                draw.multiline_text(
                    (x + 10, y_box + 70),
                    body,
                    fill="black",
                    font=small_font,
                    spacing=3,
                )

        path = png_dir / f"page_{page_idx+1:02d}.png"
        img.save(path)
        print(f"[png] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset-dir", type=Path, default=Path("bfcl_cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("ablation_4way_outputs"))
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--start", type=int, default=1, help="1-based sample start")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    samples = load_bfcl_samples(
        dataset_dir=args.dataset_dir,
        n=args.n,
        seed=args.seed,
    )

    with (args.out_dir / "samples.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    records = []

    jsonl_path = args.out_dir / "results.jsonl"

    # Resume support: read existing successful sample IDs.
    done = set()
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    old = json.loads(line)
                    sid = old["sample"]["sample_id"]
                    result_map = old.get("results", {})
                    all_ok = (
                        all(
                            result_map.get(s, {}).get("ok") is True
                            for s in "ABCD"
                        )
                    )
                    # Only successful 4-way samples are resumable.
                    # Failed samples are rerun after code/schema fixes.
                    if all_ok:
                        done.add(sid)
                        records.append(old)
                except Exception:
                    pass

    for sample in samples:
        sid = sample["sample_id"]
        if sid < args.start or sid in done:
            continue

        print("\n" + "=" * 100)
        print(f"[{sid:02d}/{len(samples)}] {sample['category']} | {sample['bfcl_id']}")
        print(sample["query"])
        print("tools:", ", ".join(t["name"] for t in sample["tools"]))

        result_map = {}

        for label, runner in [
            ("A", run_A),
            ("B", run_B),
            ("C", run_C),
            ("D", run_D),
        ]:
            print(f"  -> {label}", end="", flush=True)
            try:
                r = runner(client, sample)
                result_map[label] = r
                print(
                    f"  ok={r.get('ok')} "
                    f"api={r.get('api_calls')} "
                    f"lat={r.get('latency_total_s',0):.2f}s "
                    f"tokens={total_tokens_for_result(r)}"
                )
            except Exception as e:
                traceback.print_exc()
                result_map[label] = {
                    "strategy": label,
                    "name": "exception",
                    "ok": False,
                    "error": repr(e),
                    "pre_text": "",
                    "calls": [],
                    "observations": [],
                    "next_action": "",
                    "latency_first_s": 0.0,
                    "latency_second_s": 0.0,
                    "latency_total_s": 0.0,
                    "api_calls": 0,
                    "usage_first": {},
                    "usage_second": {},
                }

        rec = {
            "sample": sample,
            "results": result_map,
        }

        if args.judge:
            try:
                rec["judge"] = judge_four(client, sample, result_map)
            except Exception as e:
                rec["judge"] = {"error": repr(e)}

        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        records.append(rec)

    # Deduplicate if resumed.
    by_id = {}
    for r in records:
        by_id[r["sample"]["sample_id"]] = r
    records = [by_id[k] for k in sorted(by_id)]

    # CSV
    flat = [flatten_row(r) for r in records]
    csv_path = args.out_dir / "results.csv"
    if flat:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)

    # Summary CSV
    summary = summarize(records)
    summary_path = args.out_dir / "summary.csv"
    if summary:
        with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    # PNG
    render_png_pages(records, args.out_dir, per_page=5)

    print("\nDONE")
    print("samples :", args.out_dir / "samples.json")
    print("jsonl   :", jsonl_path)
    print("csv     :", csv_path)
    print("summary :", summary_path)
    print("png dir :", args.out_dir / "png")


if __name__ == "__main__":
    main()
