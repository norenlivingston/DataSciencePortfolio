"""
AI Agent — ML Pipeline Assistant
A tool-calling agent using Ollama (free, local) that can answer natural
language questions about the trained model by invoking the same tools
exposed by the MCP server.

Requirements:
    ollama pull qwen2.5        # recommended — strong tool calling support
    ollama pull llama3.2       # alternative

Usage (from projects/ directory):
    python 05_agents/agent.py
    python 05_agents/agent.py --question "Which model was selected?"
    python 05_agents/agent.py --question "Predict with all features set to 1.0"
"""
import argparse
import json
from pathlib import Path

import joblib
import ollama
import pandas as pd
import yaml


# ── Config & model ────────────────────────────────────────────────────────────

def _load_config() -> dict:
    for search in [Path.cwd(), Path.cwd().parent]:
        p = search / "config.yaml"
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found. Run from projects/ directory.")


_config   = _load_config()
_model    = joblib.load(_config["mlops"]["model_path"])
_features = list(_model.named_steps["preprocessor"].feature_names_in_)


# ── Tool implementations ──────────────────────────────────────────────────────

def list_features() -> list:
    return _features


def get_latest_metrics() -> dict:
    log_path = Path(_config["mlops"]["metrics_log"])
    if not log_path.exists():
        return {"error": "No metrics log found. Run the pipeline first."}
    history = json.loads(log_path.read_text())
    return history[-1]


def get_run_history() -> list:
    log_path = Path(_config["mlops"]["metrics_log"])
    if not log_path.exists():
        return []
    return json.loads(log_path.read_text())


def predict(features: dict) -> dict:
    missing = set(_features) - set(features)
    if missing:
        return {"error": f"Missing features: {sorted(missing)}"}
    X    = pd.DataFrame([{f: features[f] for f in _features}])
    pred = float(_model.predict(X)[0])
    return {"prediction": pred}


TOOL_REGISTRY = {
    "list_features":      lambda args: list_features(),
    "get_latest_metrics": lambda args: get_latest_metrics(),
    "get_run_history":    lambda args: get_run_history(),
    "predict":            lambda args: predict(args["features"]),
}


# ── Tool schemas (Ollama / OpenAI-compatible format) ──────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_features",
            "description": "Return the feature names required to make a prediction.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_metrics",
            "description": (
                "Return performance metrics from the most recent pipeline run: "
                "best model name, CV R², test R², RMSE, MAE, and timestamp."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_run_history",
            "description": "Return metrics from all previous pipeline runs in chronological order.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict",
            "description": (
                "Make a regression prediction given feature values. "
                "Call list_features first if you don't know the required feature names."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "features": {
                        "type": "object",
                        "description": "Dict of {feature_name: float_value} for all required features.",
                    }
                },
                "required": ["features"],
            },
        },
    },
]


# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful assistant for an ML pipeline project. "
    "You have tools to inspect a trained regression model and run predictions. "
    "Always use tools to answer accurately rather than guessing."
)


def run_agent(question: str, model: str = "qwen2.5", verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    while True:
        response = ollama.chat(model=model, messages=messages, tools=TOOLS)
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            name   = call.function.name
            args   = call.function.arguments or {}

            if verbose:
                print(f"  [tool]   {name}({json.dumps(args) if args else ''})")

            result = TOOL_REGISTRY[name](args)

            if verbose:
                print(f"  [result] {json.dumps(result, indent=None)}\n")

            messages.append({"role": "tool", "content": json.dumps(result)})


# ── Entry point ───────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "How did the model perform on the test set?",
    "Which model was selected and what does the CV score tell us?",
    "What features does the model need? Make a prediction with all of them set to 1.0.",
]


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline AI Agent")
    parser.add_argument("--question", "-q", default=None, help="Ask the agent a question")
    parser.add_argument("--model",    "-m", default="qwen2.5", help="Ollama model to use")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEMO_QUESTIONS

    for question in questions:
        print(f"\n{'─' * 55}")
        print(f"Q: {question}")
        print(f"{'─' * 55}")
        answer = run_agent(question, model=args.model)
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
