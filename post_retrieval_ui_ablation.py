#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPT-6 Luna: Post-Retrieval UI Orchestration 4-Way Ablation
=========================================================

What this experiment actually tests
-----------------------------------
The initial DATA/SEARCH tool call is NOT the ablation variable.

For every sample:

    USER
      -> LLM chooses/calls DATA TOOL
      -> deterministic mock DATA TOOL executes
      -> SAME DATA TOOL RESULT is frozen
      -> 4 orchestration branches start here

We compare how the final UI response is produced from that SAME tool result.

A. ONE_SHOT_INTERLEAVED
   data result
     -> one LLM response with actual `render_widget` tool available
     -> attempt: text_before -> widget_call -> text_after
   No second LLM call.
   This directly tests whether the model/API naturally emits post-widget text
   in the SAME generation after a UI-only function call.

B. WIDGET_THEN_CONTINUATION
   data result
     -> LLM #1: natural pre-widget text + `render_widget` call
     -> app renders widget and returns a tiny widget ack
     -> LLM #2: natural continuation after the widget
   This is the explicit:
     answer -> widget request/render -> continuation
   architecture.

C. SEPARATE_WIDGET_PLANNER
   data result
     -> LLM #1: pre-widget answer text
     -> LLM #2: UI planner only, chooses `render_widget`
     -> widget render/ack
     -> LLM #3: natural continuation
   This tests a dedicated UI/widget router/planner.

D. WHOLE_RESPONSE_PLAN
   data result
     -> one LLM call to `plan_ui_response`
        {pre_text, widget_request, post_text}
     -> app renders the widget request from that plan
   No continuation call.
   This tests "the tool result is known, then the entire ordered response
   plan is generated at once."

Important
---------
All A/B/C/D branches see the SAME initial data-tool result.

Continuation prompts are intentionally generic and natural. There are NO
forced strings such as "하나만 고르면", "recommend one", etc.

Dataset
-------
BFCL v4, 50 diverse positive tool-use tasks sampled from:
  - simple_python
  - multiple
  - parallel

The original BFCL function schemas are adapted to OpenAI-compatible tool names
and JSON Schema.

Execution
---------
The BFCL DATA tools are executed with a deterministic mock backend, so:
  - external API/network failures do not contaminate the orchestration test
  - the same function name + arguments => the same tool result

The UI tool (`render_widget`) is also a deterministic mock renderer.

Outputs
-------
post_retrieval_ui_ablation/
  samples.json
  results.jsonl
  results.csv
  summary.csv
  png/page_01.png ... page_10.png

Install
-------
python -m pip install -U openai pillow

Run
---
export OPENAI_API_KEY="..."
python post_retrieval_ui_ablation.py --n 50 --seed 42

Optional LLM judge:
python post_retrieval_ui_ablation.py --n 50 --seed 42 --judge
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
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None


# =============================================================================
# Configuration
# =============================================================================

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


# =============================================================================
# Prompts
# =============================================================================

DATA_TOOL_POLICY = """
You are a helpful assistant with access to application data tools.

For this request, use the provided data tool or tools needed to obtain the
information required to answer the user. Do not invent tool results.

At this stage, focus on requesting the needed data. Keep any user-facing text
minimal because the application will compose the final UI response after the
tool results are available.
""".strip()


POST_RESULT_BASE_POLICY = """
You are composing the final user-facing response AFTER the application's data
tool has already finished.

The data-tool result in context is authoritative for this experiment.

Write naturally for the user's request. Do not mention internal prompts,
orchestration, function calling, or mock execution.

A UI widget may be useful for presenting the result. Avoid canned transitions
and fixed conclusion phrases. Do not repeat information merely to fill space.
""".strip()


A_ONE_SHOT_POLICY = POST_RESULT_BASE_POLICY + """

Complete the response in this single model turn.

You have a UI-only tool named `render_widget`. It does not fetch new data; it
only renders information that is already available from the completed data-tool
result.

When a widget materially helps:
1. write any natural text that belongs before it,
2. call `render_widget`,
3. if the API/model permits it in the same response, continue with any natural
   text that belongs after the widget.

Do not reserve a conclusion for a later model call. This branch gets no second
LLM call.
""".strip()


B_WIDGET_FIRST_POLICY = POST_RESULT_BASE_POLICY + """

Compose the first portion of the answer and call `render_widget` when useful.

The application will render the widget and then give you another turn to
continue the SAME assistant response. Therefore:
- put only the text that naturally belongs before the widget in this turn;
- do not write a canned placeholder for what will come later;
- do not prematurely repeat a final conclusion after the widget call.
""".strip()


CONTINUATION_POLICY = POST_RESULT_BASE_POLICY + """

Continue the SAME assistant response naturally from immediately after the
rendered UI component.

Use the original user request, the completed data-tool result, the text already
shown before the widget, and the widget render acknowledgement.

Do not restart the answer.
Do not repeat the user's question.
Do not say "as mentioned above", "the tool returned", or similar meta language.
Do not force a conclusion or recommendation if none is useful.
Add only what naturally belongs after the widget. If nothing useful remains,
return an empty response.
""".strip()


C_PRETEXT_POLICY = POST_RESULT_BASE_POLICY + """

Write only the natural user-facing text that should appear BEFORE a possible
UI widget.

Do not describe widget mechanics.
Do not add a final post-widget conclusion yet.
The application has a separate UI planner that runs next.
""".strip()


C_WIDGET_PLANNER_POLICY = """
You are a UI planner.

Given:
- the user's request,
- the completed data-tool result,
- the assistant text already shown,

decide whether a UI component would materially improve the response.

If useful, call `render_widget` exactly once with the best widget request.
Do not write user-facing prose.
Do not fetch new data.
Use only information already present in the supplied data-tool result.

If a widget is not useful, call `render_widget` with widget_type="none".
""".strip()


D_PLAN_POLICY = POST_RESULT_BASE_POLICY + """

Create the entire ordered UI response plan in one model turn by calling
`plan_ui_response`.

The plan contains:
- pre_text: natural text before the UI component
- widget_request: the UI component to render from the completed data result
- post_text: natural text after the UI component

The data tool has ALREADY finished, so both pre_text and post_text may be
grounded in its actual result.

Do not use canned wording. If no widget helps, set widget_type to "none".
If no pre-text or post-text is needed, use an empty string.
""".strip()


JUDGE_POLICY = """
You are evaluating four UI orchestration strategies for the same user request
and the same completed data-tool result.

Score A/B/C/D from 1 to 5 on:
- answer_helpfulness
- grounding_to_data_result
- natural_text_widget_text_flow
- non_redundancy
- completion_of_user_request

Do not reward verbosity.
Do not penalize a strategy merely for using fewer model calls.

Return JSON only:
{
  "A": {...},
  "B": {...},
  "C": {...},
  "D": {...},
  "best": "A|B|C|D|tie",
  "reason": "short explanation"
}
""".strip()


# =============================================================================
# UI tool schemas
# =============================================================================

WIDGET_TYPES = [
    "map",
    "table",
    "cards",
    "chart",
    "status",
    "result",
    "none",
]

RENDER_WIDGET_TOOL = {
    "type": "function",
    "name": "render_widget",
    "description": (
        "Render a UI component from data that has already been retrieved. "
        "This tool does not fetch new information."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "widget_type": {
                "type": "string",
                "enum": WIDGET_TYPES,
                "description": "Best UI representation for the available result.",
            },
            "title": {
                "type": "string",
                "description": "Short optional title for the UI component.",
            },
            "data": {
                "type": "object",
                "description": (
                    "Compact data already supported by the completed data-tool "
                    "result. Do not invent new facts."
                ),
                "additionalProperties": True,
            },
        },
        "required": ["widget_type", "title", "data"],
        "additionalProperties": False,
    },
    "strict": False,
}


PLAN_UI_RESPONSE_TOOL = {
    "type": "function",
    "name": "plan_ui_response",
    "description": (
        "Plan the complete ordered user-facing response after data retrieval "
        "has already completed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pre_text": {
                "type": "string",
                "description": "Natural text shown before the UI component.",
            },
            "widget_request": {
                "type": "object",
                "properties": {
                    "widget_type": {
                        "type": "string",
                        "enum": WIDGET_TYPES,
                    },
                    "title": {"type": "string"},
                    "data": {
                        "type": "object",
                        "additionalProperties": True,
                    },
                },
                "required": ["widget_type", "title", "data"],
                "additionalProperties": False,
            },
            "post_text": {
                "type": "string",
                "description": "Natural text shown after the UI component.",
            },
        },
        "required": ["pre_text", "widget_request", "post_text"],
        "additionalProperties": False,
    },
    "strict": False,
}


# =============================================================================
# Generic utilities
# =============================================================================

def now() -> float:
    return time.perf_counter()


def safe_json(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def stable_int(obj: Any) -> int:
    raw = safe_json(obj).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:12], 16)


def model_dump_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    return str(obj)


def get_usage(response: Any) -> dict:
    usage = getattr(response, "usage", None)
    dumped = model_dump_safe(usage)
    return dumped if isinstance(dumped, dict) else {}


def usage_total_tokens(usage: dict) -> int:
    if isinstance(usage.get("total_tokens"), int):
        return usage["total_tokens"]
    return int(usage.get("input_tokens", 0) or 0) + int(
        usage.get("output_tokens", 0) or 0
    )


# =============================================================================
# BFCL loading / adaptation
# =============================================================================

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
            elif isinstance(x.get("content"), str):
                texts.append(x["content"])

    rec(q)

    texts = [x.strip() for x in texts if x and x.strip()]
    if not texts:
        return safe_json(q)

    return max(texts, key=len)


def sanitize_tool_name(raw_name: Any) -> str:
    raw = str(raw_name or "tool").strip()
    safe = OPENAI_NAME_RE.sub("_", raw)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return (safe or "tool")[:60]


def sanitize_schema_node(node: Any) -> Any:
    if isinstance(node, list):
        return [sanitize_schema_node(x) for x in node]

    if not isinstance(node, dict):
        return copy.deepcopy(node)

    out: Dict[str, Any] = {}

    for key, value in node.items():
        if key == "type":
            if isinstance(value, str):
                mapped = BFCL_TYPE_MAP.get(value.lower())
                if mapped:
                    out["type"] = mapped
            elif isinstance(value, list):
                mapped_list = []
                for t in value:
                    if isinstance(t, str):
                        mt = BFCL_TYPE_MAP.get(t.lower())
                        if mt and mt not in mapped_list:
                            mapped_list.append(mt)
                if mapped_list:
                    out["type"] = mapped_list
            continue

        if key == "properties":
            if isinstance(value, dict):
                out["properties"] = {
                    str(k): sanitize_schema_node(v)
                    for k, v in value.items()
                }
            else:
                out["properties"] = {}
            continue

        if key == "items":
            out["items"] = sanitize_schema_node(value)
            continue

        out[key] = sanitize_schema_node(value)

    if "properties" in out and "type" not in out:
        out["type"] = "object"

    if "items" in out and "type" not in out:
        out["type"] = "array"

    if out.get("type") == "object":
        props = out.get("properties", {})
        if not isinstance(props, dict):
            props = {}
        out["properties"] = props

        required = out.get("required", [])
        if not isinstance(required, list):
            required = []

        out["required"] = [
            str(x)
            for x in required
            if str(x) in props
        ]
        out.setdefault("additionalProperties", False)

    return out


def sanitize_parameters(params: Any) -> dict:
    if not isinstance(params, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    out = sanitize_schema_node(copy.deepcopy(params))

    if not isinstance(out, dict):
        out = {}

    out["type"] = "object"
    out.setdefault("properties", {})
    if not isinstance(out["properties"], dict):
        out["properties"] = {}

    required = out.get("required", [])
    if not isinstance(required, list):
        required = []

    out["required"] = [
        str(x)
        for x in required
        if str(x) in out["properties"]
    ]
    out.setdefault("additionalProperties", False)

    return out


def normalize_tool(fn: dict) -> dict:
    raw_name = fn.get("name") or fn.get("function_name") or "tool"
    name = sanitize_tool_name(raw_name)

    desc = fn.get("description") or f"Call {raw_name}."
    if str(raw_name) != name:
        desc = f"Original BFCL function: {raw_name}. {desc}"

    return {
        "type": "function",
        "name": name,
        "description": str(desc)[:4000],
        "parameters": sanitize_parameters(fn.get("parameters", {})),
        "strict": False,
    }


def tool_names(row: dict) -> List[str]:
    return [
        str(x.get("name", ""))
        for x in row.get("function", [])
    ]


def select_diverse(
    rows_by_cat: Dict[str, List[dict]],
    n: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    categories = list(rows_by_cat.keys())

    if n == 50 and set(categories) >= {
        "simple_python",
        "multiple",
        "parallel",
    }:
        quotas = {
            "simple_python": 20,
            "multiple": 15,
            "parallel": 15,
        }
    else:
        base = n // len(categories)
        quotas = {c: base for c in categories}
        for c in categories[: n - base * len(categories)]:
            quotas[c] += 1

    selected: List[dict] = []
    seen_names = set()

    for cat in categories:
        rows = list(rows_by_cat[cat])
        rng.shuffle(rows)

        quota = quotas.get(cat, 0)
        chosen = []

        for row in rows:
            names = set(tool_names(row))
            if names - seen_names:
                chosen.append(row)
                seen_names.update(names)

            if len(chosen) >= quota:
                break

        if len(chosen) < quota:
            chosen_ids = {id(x) for x in chosen}
            for row in rows:
                if id(row) in chosen_ids:
                    continue
                chosen.append(row)
                if len(chosen) >= quota:
                    break

        for row in chosen[:quota]:
            item = copy.deepcopy(row)
            item["_category"] = cat
            selected.append(item)

    rng.shuffle(selected)
    return selected[:n]


def load_bfcl_samples(
    dataset_dir: Path,
    n: int,
    seed: int,
) -> List[dict]:
    rows_by_cat: Dict[str, List[dict]] = {}

    for cat, filename in BFCL_FILES.items():
        path = dataset_dir / filename
        download(f"{BFCL_BASE}/{filename}", path)
        rows_by_cat[cat] = load_jsonl(path)

    raw = select_diverse(
        rows_by_cat,
        n=n,
        seed=seed,
    )

    samples = []

    for idx, row in enumerate(raw, 1):
        tools = [
            normalize_tool(fn)
            for fn in row.get("function", [])
        ]

        if not tools:
            continue

        # collision-safe tool names
        seen: Dict[str, int] = {}
        for tool in tools:
            base = tool["name"]
            count = seen.get(base, 0)
            if count:
                suffix = hashlib.sha1(
                    (
                        base
                        + "|"
                        + tool.get("description", "")
                    ).encode("utf-8")
                ).hexdigest()[:6]
                tool["name"] = f"{base[:53]}_{suffix}"
            seen[base] = count + 1

        samples.append(
            {
                "sample_id": idx,
                "bfcl_id": row.get("id"),
                "category": row.get("_category"),
                "query": flatten_question(row.get("question")),
                "data_tools": tools,
            }
        )

    return samples[:n]


# =============================================================================
# Deterministic DATA tool backend
# =============================================================================

def mock_execute_data_tool(name: str, args: dict) -> dict:
    """
    Deterministic mock backend.

    Same function name + args => same result.
    No side effects are performed.
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

    # Search / retrieval / nearby / lookup
    if any(
        token in key
        for token in [
            "search",
            "find",
            "lookup",
            "query",
            "list",
            "retrieve",
            "nearby",
            "restaurant",
            "hotel",
            "store",
        ]
    ):
        base["results"] = [
            {
                "id": f"r{i + 1}",
                "name": f"Result {i + 1}",
                "score": round(
                    0.92 - i * 0.07 + rng.uniform(-0.01, 0.01),
                    3,
                ),
                "summary": f"Relevant mock result {i + 1} for {name}",
            }
            for i in range(3)
        ]
        return base

    # Weather / environment
    if any(
        token in key
        for token in [
            "weather",
            "forecast",
            "temperature",
            "air_quality",
            "climate",
        ]
    ):
        base["result"] = {
            "condition": rng.choice(
                ["Clear", "Cloudy", "Light rain"]
            ),
            "temperature_c": rng.randint(8, 29),
            "humidity_pct": rng.randint(35, 82),
            "index": rng.randint(20, 120),
        }
        return base

    # Market / finance / price
    if any(
        token in key
        for token in [
            "stock",
            "market",
            "price",
            "quote",
            "currency",
            "exchange",
            "dividend",
            "finance",
        ]
    ):
        base["result"] = {
            "value": round(
                50 + (h % 10000) / 137.0,
                2,
            ),
            "change_pct": round(
                rng.uniform(-3.0, 3.0),
                2,
            ),
            "currency": "USD",
        }
        return base

    # Booking / scheduling / update side effects
    if any(
        token in key
        for token in [
            "book",
            "reservation",
            "appointment",
            "update",
            "create",
            "delete",
            "insert",
            "send",
        ]
    ):
        base["result"] = {
            "status": "simulated_success",
            "confirmation_id": f"mock-{h % 100000:05d}",
        }
        return base

    # Numeric / calculator-like
    numeric_values = [
        float(v)
        for v in args.values()
        if isinstance(v, (int, float))
    ]

    if any(
        token in key
        for token in [
            "calculate",
            "compute",
            "math",
            "factorial",
            "probability",
            "circumference",
            "gradient",
            "capacity",
            "field",
            "similarity",
        ]
    ):
        base["result"] = {
            "numeric_result": (
                round(sum(numeric_values), 6)
                if numeric_values
                else round((h % 100000) / 97.0, 6)
            )
        }
        return base

    base["result"] = {
        "status": "success",
        "value": f"mock_value_{h % 10000:04d}",
        "details": f"Deterministic mock result for {name}",
    }
    return base


# =============================================================================
# Model-output parsing
# =============================================================================

def parse_arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw

    if not isinstance(raw, str):
        return {}

    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {"__raw__": raw}


def extract_text_from_message_item(item: Any) -> str:
    """
    Robustly extract output_text from a Responses API message item.
    """

    dumped = model_dump_safe(item)
    if not isinstance(dumped, dict):
        return ""

    content = dumped.get("content", [])
    texts: List[str] = []

    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue

            typ = part.get("type")
            if typ in ("output_text", "text"):
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)

    return "".join(texts).strip()


def parse_output_sequence(response: Any) -> List[dict]:
    """
    Preserve response.output ordering.

    Each element:
      {"type":"text", ...}
      {"type":"function_call", ...}
      {"type":"other", ...}
    """

    seq: List[dict] = []

    for item in getattr(response, "output", []) or []:
        typ = getattr(item, "type", None)

        if typ == "message":
            text = extract_text_from_message_item(item)
            if text:
                seq.append(
                    {
                        "type": "text",
                        "text": text,
                    }
                )

        elif typ == "function_call":
            seq.append(
                {
                    "type": "function_call",
                    "name": getattr(item, "name", None),
                    "call_id": getattr(item, "call_id", None),
                    "arguments": parse_arguments(
                        getattr(item, "arguments", "{}")
                    ),
                }
            )

        else:
            seq.append(
                {
                    "type": "other",
                    "raw": model_dump_safe(item),
                }
            )

    # Fallback: if SDK exposes output_text but message parsing missed it.
    if (
        not any(x["type"] == "text" for x in seq)
        and getattr(response, "output_text", "")
    ):
        seq.append(
            {
                "type": "text",
                "text": response.output_text,
            }
        )

    return seq


def collect_function_calls(
    seq: List[dict],
    name: Optional[str] = None,
) -> List[dict]:
    out = []

    for item in seq:
        if item.get("type") != "function_call":
            continue

        if name is not None and item.get("name") != name:
            continue

        out.append(item)

    return out


def split_text_around_first_widget(
    seq: List[dict],
) -> Tuple[str, Optional[dict], str]:
    """
    Preserve ordering to test whether text appears after a widget call in
    the SAME model response.
    """

    before: List[str] = []
    after: List[str] = []
    widget_call: Optional[dict] = None
    seen_widget = False

    for item in seq:
        if (
            item.get("type") == "function_call"
            and item.get("name") == "render_widget"
            and widget_call is None
        ):
            widget_call = item
            seen_widget = True
            continue

        if item.get("type") == "text":
            if seen_widget:
                after.append(item.get("text", ""))
            else:
                before.append(item.get("text", ""))

    return (
        "\n".join(x for x in before if x).strip(),
        widget_call,
        "\n".join(x for x in after if x).strip(),
    )


# =============================================================================
# OpenAI API helper
# =============================================================================

def call_model(
    client: OpenAI,
    *,
    instructions: str,
    input_items: Any,
    tools: Optional[List[dict]] = None,
    tool_choice: Any = None,
    previous_response_id: Optional[str] = None,
) -> Tuple[Any, float]:
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "reasoning": {"effort": "none"},
        "instructions": instructions,
        "input": input_items,
    }

    if tools is not None:
        kwargs["tools"] = tools

    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice

    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id

    start = now()
    response = client.responses.create(**kwargs)
    latency = now() - start

    return response, latency


# =============================================================================
# Common initial DATA-tool stage
# =============================================================================

def run_common_data_stage(
    client: OpenAI,
    sample: dict,
) -> dict:
    """
    This is performed ONCE per sample and reused by A/B/C/D.
    """

    r1, latency = call_model(
        client,
        instructions=DATA_TOOL_POLICY,
        input_items=[
            {
                "role": "user",
                "content": sample["query"],
            }
        ],
        tools=sample["data_tools"],
        tool_choice="required",
    )

    seq = parse_output_sequence(r1)

    calls = [
        item
        for item in seq
        if item.get("type") == "function_call"
    ]

    if not calls:
        raise RuntimeError(
            "Common data stage produced no function_call."
        )

    observations = []

    for call in calls:
        result = mock_execute_data_tool(
            call["name"],
            call["arguments"],
        )

        observations.append(
            {
                "call_id": call["call_id"],
                "name": call["name"],
                "arguments": call["arguments"],
                "result": result,
            }
        )

    function_outputs = [
        {
            "type": "function_call_output",
            "call_id": obs["call_id"],
            "output": json.dumps(
                obs["result"],
                ensure_ascii=False,
            ),
        }
        for obs in observations
    ]

    return {
        "response_id": r1.id,
        "response_sequence": seq,
        "calls": calls,
        "observations": observations,
        "function_outputs": function_outputs,
        "latency_s": latency,
        "usage": get_usage(r1),
    }


# =============================================================================
# Widget execution
# =============================================================================

def execute_widget_request(
    widget_request: Optional[dict],
) -> dict:
    """
    UI-only deterministic renderer.

    It intentionally returns only a render acknowledgement.
    It does NOT add new factual data.
    """

    if not widget_request:
        return {
            "rendered": False,
            "widget_type": "none",
        }

    widget_type = widget_request.get(
        "widget_type",
        "none",
    )

    if widget_type == "none":
        return {
            "rendered": False,
            "widget_type": "none",
        }

    payload = {
        "widget_type": widget_type,
        "title": widget_request.get("title", ""),
        "data": widget_request.get("data", {}),
    }

    return {
        "rendered": True,
        "widget_type": widget_type,
        "widget_id": (
            "widget_"
            + hashlib.sha1(
                safe_json(payload).encode("utf-8")
            ).hexdigest()[:10]
        ),
    }


def widget_output_item(
    widget_call: dict,
    widget_ack: dict,
) -> dict:
    return {
        "type": "function_call_output",
        "call_id": widget_call["call_id"],
        "output": json.dumps(
            widget_ack,
            ensure_ascii=False,
        ),
    }


# =============================================================================
# Context helpers after DATA result
# =============================================================================

def common_post_result_context(
    sample: dict,
    common: dict,
) -> List[dict]:
    """
    Explicit standalone context representing the state AFTER the common
    data tool has completed.

    We use this for branches that do not rely on previous_response_id.
    """

    return [
        {
            "role": "user",
            "content": sample["query"],
        },
        {
            "role": "assistant",
            "content": (
                "The application data tool has completed. "
                "Use the following completed result as context."
            ),
        },
        {
            "role": "user",
            "content": (
                "COMPLETED_DATA_TOOL_RESULT\n"
                + json.dumps(
                    common["observations"],
                    ensure_ascii=False,
                )
            ),
        },
    ]


# =============================================================================
# Strategy A
# ONE_SHOT_INTERLEAVED
# =============================================================================

def run_A(
    client: OpenAI,
    sample: dict,
    common: dict,
) -> dict:
    start = now()

    r, latency = call_model(
        client,
        instructions=A_ONE_SHOT_POLICY,
        previous_response_id=common["response_id"],
        input_items=common["function_outputs"],
        tools=[RENDER_WIDGET_TOOL],
        tool_choice="auto",
    )

    seq = parse_output_sequence(r)

    pre_text, widget_call, post_text = (
        split_text_around_first_widget(seq)
    )

    widget_ack = None
    if widget_call:
        widget_ack = execute_widget_request(
            widget_call["arguments"]
        )

    return {
        "strategy": "A",
        "name": "one_shot_interleaved",
        "ok": True,
        "pre_text": pre_text,
        "widget_call": widget_call,
        "widget_ack": widget_ack,
        "post_text": post_text,
        "raw_sequence": seq,
        "post_text_same_generation": bool(post_text),
        "api_calls_after_data": 1,
        "latency_after_data_s": now() - start,
        "latency_calls": [latency],
        "usage": [get_usage(r)],
    }


# =============================================================================
# Strategy B
# WIDGET_THEN_CONTINUATION
# =============================================================================

def run_B(
    client: OpenAI,
    sample: dict,
    common: dict,
) -> dict:
    start = now()

    # First post-result turn: pre-text + widget request.
    r1, l1 = call_model(
        client,
        instructions=B_WIDGET_FIRST_POLICY,
        previous_response_id=common["response_id"],
        input_items=common["function_outputs"],
        tools=[RENDER_WIDGET_TOOL],
        tool_choice="auto",
    )

    seq1 = parse_output_sequence(r1)

    pre_text, widget_call, accidental_post_text = (
        split_text_around_first_widget(seq1)
    )

    if widget_call:
        widget_ack = execute_widget_request(
            widget_call["arguments"]
        )

        continuation_input = [
            widget_output_item(
                widget_call,
                widget_ack,
            )
        ]

    else:
        widget_ack = {
            "rendered": False,
            "widget_type": "none",
        }

        # No widget call: still allow a natural continuation from the
        # first response by passing a compact state message.
        continuation_input = [
            {
                "role": "user",
                "content": (
                    "No UI component was rendered. Continue only if "
                    "something naturally remains to complete the answer."
                ),
            }
        ]

    r2, l2 = call_model(
        client,
        instructions=CONTINUATION_POLICY,
        previous_response_id=r1.id,
        input_items=continuation_input,
        tools=[RENDER_WIDGET_TOOL],
        tool_choice="none",
    )

    seq2 = parse_output_sequence(r2)

    continuation_text = "\n".join(
        item["text"]
        for item in seq2
        if item.get("type") == "text"
    ).strip()

    post_text = "\n".join(
        x
        for x in [
            accidental_post_text,
            continuation_text,
        ]
        if x
    ).strip()

    return {
        "strategy": "B",
        "name": "widget_then_continuation",
        "ok": True,
        "pre_text": pre_text,
        "widget_call": widget_call,
        "widget_ack": widget_ack,
        "post_text": post_text,
        "raw_sequence_first": seq1,
        "raw_sequence_second": seq2,
        "accidental_post_text_before_continuation": (
            accidental_post_text
        ),
        "api_calls_after_data": 2,
        "latency_after_data_s": now() - start,
        "latency_calls": [l1, l2],
        "usage": [
            get_usage(r1),
            get_usage(r2),
        ],
    }


# =============================================================================
# Strategy C
# SEPARATE_WIDGET_PLANNER
# =============================================================================

def run_C(
    client: OpenAI,
    sample: dict,
    common: dict,
) -> dict:
    start = now()

    base_context = common_post_result_context(
        sample,
        common,
    )

    # 1) Natural pre-widget answer text.
    r1, l1 = call_model(
        client,
        instructions=C_PRETEXT_POLICY,
        input_items=base_context,
        tools=None,
        tool_choice=None,
    )

    seq1 = parse_output_sequence(r1)

    pre_text = "\n".join(
        item["text"]
        for item in seq1
        if item.get("type") == "text"
    ).strip()

    # 2) Dedicated widget planner.
    planner_context = base_context + [
        {
            "role": "assistant",
            "content": pre_text,
        },
        {
            "role": "user",
            "content": (
                "Choose the UI component, if any, for the response "
                "state above."
            ),
        },
    ]

    r2, l2 = call_model(
        client,
        instructions=C_WIDGET_PLANNER_POLICY,
        input_items=planner_context,
        tools=[RENDER_WIDGET_TOOL],
        tool_choice="required",
    )

    seq2 = parse_output_sequence(r2)

    widget_calls = collect_function_calls(
        seq2,
        "render_widget",
    )

    widget_call = (
        widget_calls[0]
        if widget_calls
        else None
    )

    widget_ack = execute_widget_request(
        widget_call["arguments"]
        if widget_call
        else {
            "widget_type": "none",
            "title": "",
            "data": {},
        }
    )

    # 3) Natural post-widget continuation.
    continuation_context = planner_context + [
        {
            "role": "assistant",
            "content": (
                "UI_RENDER_STATE\n"
                + json.dumps(
                    {
                        "widget_request": (
                            widget_call["arguments"]
                            if widget_call
                            else None
                        ),
                        "widget_ack": widget_ack,
                    },
                    ensure_ascii=False,
                )
            ),
        }
    ]

    r3, l3 = call_model(
        client,
        instructions=CONTINUATION_POLICY,
        input_items=continuation_context,
        tools=None,
        tool_choice=None,
    )

    seq3 = parse_output_sequence(r3)

    post_text = "\n".join(
        item["text"]
        for item in seq3
        if item.get("type") == "text"
    ).strip()

    return {
        "strategy": "C",
        "name": "separate_widget_planner",
        "ok": True,
        "pre_text": pre_text,
        "widget_call": widget_call,
        "widget_ack": widget_ack,
        "post_text": post_text,
        "raw_sequence_pretext": seq1,
        "raw_sequence_planner": seq2,
        "raw_sequence_continuation": seq3,
        "api_calls_after_data": 3,
        "latency_after_data_s": now() - start,
        "latency_calls": [l1, l2, l3],
        "usage": [
            get_usage(r1),
            get_usage(r2),
            get_usage(r3),
        ],
    }


# =============================================================================
# Strategy D
# WHOLE_RESPONSE_PLAN
# =============================================================================

def run_D(
    client: OpenAI,
    sample: dict,
    common: dict,
) -> dict:
    start = now()

    r, latency = call_model(
        client,
        instructions=D_PLAN_POLICY,
        previous_response_id=common["response_id"],
        input_items=common["function_outputs"],
        tools=[PLAN_UI_RESPONSE_TOOL],
        tool_choice="required",
    )

    seq = parse_output_sequence(r)

    plan_calls = collect_function_calls(
        seq,
        "plan_ui_response",
    )

    if not plan_calls:
        return {
            "strategy": "D",
            "name": "whole_response_plan",
            "ok": False,
            "error": "No plan_ui_response call",
            "pre_text": "",
            "widget_call": None,
            "widget_ack": None,
            "post_text": "",
            "raw_sequence": seq,
            "api_calls_after_data": 1,
            "latency_after_data_s": now() - start,
            "latency_calls": [latency],
            "usage": [get_usage(r)],
        }

    plan = plan_calls[0]["arguments"]

    pre_text = str(
        plan.get("pre_text", "") or ""
    ).strip()

    widget_request = plan.get(
        "widget_request",
        {},
    )
    if not isinstance(widget_request, dict):
        widget_request = {}

    post_text = str(
        plan.get("post_text", "") or ""
    ).strip()

    widget_ack = execute_widget_request(
        widget_request
    )

    pseudo_widget_call = {
        "type": "function_call",
        "name": "render_widget",
        "call_id": None,
        "arguments": widget_request,
        "planned_inside": "plan_ui_response",
    }

    return {
        "strategy": "D",
        "name": "whole_response_plan",
        "ok": True,
        "pre_text": pre_text,
        "widget_call": pseudo_widget_call,
        "widget_ack": widget_ack,
        "post_text": post_text,
        "raw_sequence": seq,
        "plan": plan,
        "api_calls_after_data": 1,
        "latency_after_data_s": now() - start,
        "latency_calls": [latency],
        "usage": [get_usage(r)],
    }


# =============================================================================
# Optional judge
# =============================================================================

def visible_render(
    sample: dict,
    common: dict,
    result: dict,
) -> str:
    widget_args = None

    if result.get("widget_call"):
        widget_args = result["widget_call"].get(
            "arguments"
        )

    return (
        f"USER:\n{sample['query']}\n\n"
        f"DATA RESULT:\n"
        f"{json.dumps(common['observations'], ensure_ascii=False)}\n\n"
        f"PRE-TEXT:\n{result.get('pre_text','')}\n\n"
        f"WIDGET REQUEST:\n"
        f"{json.dumps(widget_args, ensure_ascii=False)}\n\n"
        f"POST-TEXT:\n{result.get('post_text','')}"
    )


def judge_four(
    client: OpenAI,
    sample: dict,
    common: dict,
    results: Dict[str, dict],
) -> dict:
    payload = {
        key: visible_render(
            sample,
            common,
            results[key],
        )
        for key in "ABCD"
    }

    r, _ = call_model(
        client,
        instructions=JUDGE_POLICY,
        input_items=[
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            }
        ],
    )

    text = getattr(
        r,
        "output_text",
        "",
    ) or ""

    try:
        return json.loads(text)
    except Exception:
        return {"raw": text}


# =============================================================================
# Metrics
# =============================================================================

def result_total_tokens(result: dict) -> int:
    return sum(
        usage_total_tokens(x)
        for x in result.get("usage", [])
        if isinstance(x, dict)
    )


def text_chars(result: dict) -> int:
    return len(
        (result.get("pre_text") or "")
        + (result.get("post_text") or "")
    )


def widget_type(result: dict) -> str:
    call = result.get("widget_call")
    if not call:
        return "none"

    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return "none"

    return str(
        args.get("widget_type", "none")
    )


def flatten_record(record: dict) -> dict:
    sample = record["sample"]
    common = record["common"]

    row = {
        "sample_id": sample["sample_id"],
        "bfcl_id": sample["bfcl_id"],
        "category": sample["category"],
        "query": sample["query"],
        "common_data_latency_s": round(
            common["latency_s"],
            4,
        ),
        "common_data_tool_calls": safe_json(
            common["calls"]
        ),
        "common_data_result": safe_json(
            common["observations"]
        ),
    }

    for key in "ABCD":
        result = record["results"][key]

        row.update(
            {
                f"{key}_name": result.get(
                    "name",
                    "",
                ),
                f"{key}_ok": result.get(
                    "ok",
                    False,
                ),
                f"{key}_api_calls_after_data": (
                    result.get(
                        "api_calls_after_data",
                        0,
                    )
                ),
                f"{key}_latency_after_data_s": round(
                    result.get(
                        "latency_after_data_s",
                        0.0,
                    ),
                    4,
                ),
                f"{key}_tokens_after_data": (
                    result_total_tokens(result)
                ),
                f"{key}_text_chars": (
                    text_chars(result)
                ),
                f"{key}_widget_type": (
                    widget_type(result)
                ),
                f"{key}_pre_text": (
                    result.get(
                        "pre_text",
                        "",
                    )
                ),
                f"{key}_widget_call": safe_json(
                    result.get(
                        "widget_call"
                    )
                ),
                f"{key}_post_text": (
                    result.get(
                        "post_text",
                        "",
                    )
                ),
            }
        )

        if key == "A":
            row[
                "A_post_text_same_generation"
            ] = result.get(
                "post_text_same_generation",
                False,
            )

    if "judge" in record:
        row["judge"] = safe_json(
            record["judge"]
        )

    return row


def summarize(
    records: List[dict],
) -> List[dict]:
    rows = []

    for key in "ABCD":
        results = [
            rec["results"][key]
            for rec in records
        ]

        if not results:
            continue

        rows.append(
            {
                "strategy": key,
                "name": results[0].get(
                    "name",
                    "",
                ),
                "n": len(results),
                "success": sum(
                    bool(r.get("ok"))
                    for r in results
                ),
                "mean_api_calls_after_data": round(
                    sum(
                        r.get(
                            "api_calls_after_data",
                            0,
                        )
                        for r in results
                    )
                    / len(results),
                    3,
                ),
                "mean_latency_after_data_s": round(
                    sum(
                        r.get(
                            "latency_after_data_s",
                            0.0,
                        )
                        for r in results
                    )
                    / len(results),
                    4,
                ),
                "mean_tokens_after_data": round(
                    sum(
                        result_total_tokens(r)
                        for r in results
                    )
                    / len(results),
                    2,
                ),
                "mean_text_chars": round(
                    sum(
                        text_chars(r)
                        for r in results
                    )
                    / len(results),
                    2,
                ),
                "widget_render_rate": round(
                    sum(
                        widget_type(r)
                        != "none"
                        for r in results
                    )
                    / len(results),
                    4,
                ),
                "post_text_rate": round(
                    sum(
                        bool(
                            (
                                r.get(
                                    "post_text"
                                )
                                or ""
                            ).strip()
                        )
                        for r in results
                    )
                    / len(results),
                    4,
                ),
                "same_generation_post_text_rate": (
                    round(
                        sum(
                            bool(
                                r.get(
                                    "post_text_same_generation"
                                )
                            )
                            for r in results
                        )
                        / len(results),
                        4,
                    )
                    if key == "A"
                    else ""
                ),
            }
        )

    return rows


# =============================================================================
# PNG rendering (Pillow only)
# =============================================================================

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

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(
                    path,
                    size=size,
                )
            except Exception:
                pass

    return ImageFont.load_default()


def wrap_text(
    text: str,
    width: int,
) -> str:
    if not text:
        return ""

    lines: List[str] = []

    for para in str(text).splitlines():
        if not para:
            lines.append("")
            continue

        lines.extend(
            textwrap.wrap(
                para,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )

    return "\n".join(lines)


def compact_data_result(
    common: dict,
    max_chars: int = 750,
) -> str:
    text = safe_json(
        common["observations"]
    )

    if len(text) > max_chars:
        text = (
            text[: max_chars - 3]
            + "..."
        )

    return text


def compact_widget(
    result: dict,
    max_chars: int = 550,
) -> str:
    call = result.get("widget_call")

    if not call:
        return "(none)"

    text = safe_json(
        call.get("arguments", {})
    )

    if len(text) > max_chars:
        text = (
            text[: max_chars - 3]
            + "..."
        )

    return text


def render_png_pages(
    records: List[dict],
    out_dir: Path,
    per_page: int = 5,
) -> None:
    if Image is None:
        print(
            "[png] Pillow unavailable; "
            "skipping PNG output."
        )
        return

    png_dir = out_dir / "png"
    png_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    title_font = find_font(24)
    body_font = find_font(15)
    small_font = find_font(13)

    width = 2400
    margin = 30
    col_gap = 18
    cols = 4

    col_width = (
        width
        - 2 * margin
        - (cols - 1) * col_gap
    ) // cols

    row_height = 790
    header_height = 160

    pages = math.ceil(
        len(records) / per_page
    )

    for page_idx in range(pages):
        chunk = records[
            page_idx * per_page:
            (page_idx + 1) * per_page
        ]

        height = (
            header_height
            + row_height * len(chunk)
            + margin
        )

        img = Image.new(
            "RGB",
            (width, height),
            "white",
        )
        draw = ImageDraw.Draw(img)

        draw.text(
            (margin, 24),
            (
                "GPT-6 Luna Post-Retrieval UI "
                f"Orchestration | Page "
                f"{page_idx + 1}/{pages}"
            ),
            fill="black",
            font=title_font,
        )

        draw.text(
            (margin, 64),
            (
                "A=one-shot interleaved | "
                "B=widget then continuation | "
                "C=separate widget planner | "
                "D=whole response plan"
            ),
            fill="black",
            font=small_font,
        )

        draw.text(
            (margin, 92),
            (
                "COMMON DATA TOOL RESULT is frozen "
                "before all four branches."
            ),
            fill="black",
            font=small_font,
        )

        for row_idx, record in enumerate(
            chunk
        ):
            sample = record["sample"]
            common = record["common"]

            y0 = (
                header_height
                + row_idx * row_height
            )

            query_line = (
                f"#{sample['sample_id']:02d} "
                f"[{sample['category']}] "
                f"{sample['query']}"
            )

            draw.text(
                (margin, y0),
                wrap_text(
                    query_line,
                    150,
                ),
                fill="black",
                font=body_font,
            )

            y_box = y0 + 55

            for col_idx, key in enumerate(
                "ABCD"
            ):
                result = record[
                    "results"
                ][key]

                x = (
                    margin
                    + col_idx
                    * (
                        col_width
                        + col_gap
                    )
                )

                draw.rectangle(
                    [
                        x,
                        y_box,
                        x + col_width,
                        y_box
                        + row_height
                        - 80,
                    ],
                    outline="black",
                    width=1,
                )

                header = (
                    f"{key} | "
                    f"{result.get('name','')}\n"
                    f"api={result.get('api_calls_after_data',0)} "
                    f"lat={result.get('latency_after_data_s',0):.2f}s "
                    f"tok={result_total_tokens(result)} "
                    f"widget={widget_type(result)}"
                )

                if key == "A":
                    header += (
                        "\npost-after-widget-in-same-generation="
                        + str(
                            result.get(
                                "post_text_same_generation",
                                False,
                            )
                        )
                    )

                draw.multiline_text(
                    (
                        x + 10,
                        y_box + 10,
                    ),
                    header,
                    fill="black",
                    font=small_font,
                    spacing=3,
                )

                body = (
                    "PRE\n"
                    + (
                        result.get(
                            "pre_text",
                            "",
                        )
                        or "(empty)"
                    )
                    + "\n\nWIDGET\n"
                    + compact_widget(
                        result
                    )
                    + "\n\nPOST\n"
                    + (
                        result.get(
                            "post_text",
                            "",
                        )
                        or "(empty)"
                    )
                )

                body = wrap_text(
                    body,
                    55,
                )

                if len(body) > 3000:
                    body = (
                        body[:2997]
                        + "..."
                    )

                draw.multiline_text(
                    (
                        x + 10,
                        y_box + 90,
                    ),
                    body,
                    fill="black",
                    font=small_font,
                    spacing=3,
                )

        out_path = (
            png_dir
            / f"page_{page_idx + 1:02d}.png"
        )

        img.save(out_path)
        print(f"[png] {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("bfcl_cache"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "post_retrieval_ui_ablation"
        ),
    )
    parser.add_argument(
        "--judge",
        action="store_true",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        )
    )

    samples = load_bfcl_samples(
        args.dataset_dir,
        n=args.n,
        seed=args.seed,
    )

    with (
        args.out_dir
        / "samples.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            samples,
            f,
            ensure_ascii=False,
            indent=2,
        )

    jsonl_path = (
        args.out_dir
        / "results.jsonl"
    )

    records: List[dict] = []
    completed = set()

    # Resume only fully successful records.
    if jsonl_path.exists():
        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    result_map = rec.get(
                        "results",
                        {},
                    )

                    if all(
                        result_map.get(
                            key,
                            {},
                        ).get("ok")
                        is True
                        for key in "ABCD"
                    ):
                        sid = rec[
                            "sample"
                        ][
                            "sample_id"
                        ]
                        completed.add(sid)
                        records.append(rec)

                except Exception:
                    pass

    for sample in samples:
        sid = sample["sample_id"]

        if sid < args.start:
            continue

        if sid in completed:
            continue

        print(
            "\n"
            + "=" * 100
        )
        print(
            f"[{sid:02d}/{len(samples)}] "
            f"{sample['category']} | "
            f"{sample['bfcl_id']}"
        )
        print(sample["query"])
        print(
            "DATA tools:",
            ", ".join(
                t["name"]
                for t in sample[
                    "data_tools"
                ]
            ),
        )

        # -----------------------------------------------------
        # COMMON INITIAL DATA TOOL STAGE
        # -----------------------------------------------------
        try:
            common = run_common_data_stage(
                client,
                sample,
            )
        except Exception as exc:
            traceback.print_exc()
            print(
                "[COMMON DATA STAGE FAILED]",
                repr(exc),
            )
            continue

        print(
            "  common data stage:"
            f" calls={len(common['calls'])}"
            f" latency={common['latency_s']:.2f}s"
        )

        result_map: Dict[str, dict] = {}

        runners = [
            ("A", run_A),
            ("B", run_B),
            ("C", run_C),
            ("D", run_D),
        ]

        for key, runner in runners:
            print(
                f"  -> {key}",
                end="",
                flush=True,
            )

            try:
                result = runner(
                    client,
                    sample,
                    common,
                )

                result_map[key] = result

                print(
                    f" ok={result.get('ok')}"
                    f" api={result.get('api_calls_after_data')}"
                    f" lat={result.get('latency_after_data_s',0):.2f}s"
                    f" tok={result_total_tokens(result)}"
                    f" widget={widget_type(result)}"
                )

            except Exception as exc:
                traceback.print_exc()

                result_map[key] = {
                    "strategy": key,
                    "name": "exception",
                    "ok": False,
                    "error": repr(exc),
                    "pre_text": "",
                    "widget_call": None,
                    "widget_ack": None,
                    "post_text": "",
                    "api_calls_after_data": 0,
                    "latency_after_data_s": 0.0,
                    "latency_calls": [],
                    "usage": [],
                }

                print(
                    " ERROR",
                    repr(exc),
                )

        record = {
            "sample": sample,
            "common": common,
            "results": result_map,
        }

        if args.judge:
            try:
                record["judge"] = judge_four(
                    client,
                    sample,
                    common,
                    result_map,
                )
            except Exception as exc:
                record["judge"] = {
                    "error": repr(exc)
                }

        with jsonl_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        records.append(record)

    # Deduplicate resumed samples.
    by_id: Dict[int, dict] = {}

    for rec in records:
        sid = rec[
            "sample"
        ][
            "sample_id"
        ]
        by_id[sid] = rec

    records = [
        by_id[sid]
        for sid in sorted(by_id)
    ]

    # results.csv
    flat_rows = [
        flatten_record(rec)
        for rec in records
    ]

    csv_path = (
        args.out_dir
        / "results.csv"
    )

    if flat_rows:
        with csv_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    flat_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(flat_rows)

    # summary.csv
    summary_rows = summarize(
        records
    )

    summary_path = (
        args.out_dir
        / "summary.csv"
    )

    if summary_rows:
        with summary_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    summary_rows[0].keys()
                ),
            )
            writer.writeheader()
            writer.writerows(
                summary_rows
            )

    # PNG
    render_png_pages(
        records,
        args.out_dir,
        per_page=5,
    )

    print("\nDONE")
    print(
        "samples :",
        args.out_dir / "samples.json",
    )
    print(
        "jsonl   :",
        jsonl_path,
    )
    print(
        "csv     :",
        csv_path,
    )
    print(
        "summary :",
        summary_path,
    )
    print(
        "png dir :",
        args.out_dir / "png",
    )


if __name__ == "__main__":
    main()
