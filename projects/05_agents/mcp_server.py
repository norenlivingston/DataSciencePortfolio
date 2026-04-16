"""
MCP Server — ML Pipeline Tools
Exposes the trained regression pipeline as tools any MCP-compatible
client can call (Claude Desktop, MCP Inspector, custom agents).

Usage (from projects/ directory):
    python 05_agents/mcp_server.py

Test without any API key using MCP Inspector:
    npx @modelcontextprotocol/inspector python 05_agents/mcp_server.py

Connect to Claude Desktop — add this to your claude_desktop_config.json:
    {
      "mcpServers": {
        "ml-pipeline": {
          "command": "python",
          "args": ["/absolute/path/to/projects/05_agents/mcp_server.py"]
        }
      }
    }
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from mcp.server.fastmcp import FastMCP


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


# ── MCP Server ────────────────────────────────────────────────────────────────

mcp = FastMCP("ml-pipeline")


@mcp.tool()
def list_features() -> list:
    """Return the feature names required to make a prediction."""
    return _features


@mcp.tool()
def get_latest_metrics() -> dict:
    """Return performance metrics from the most recent pipeline run."""
    log_path = Path(_config["mlops"]["metrics_log"])
    if not log_path.exists():
        return {"error": "No metrics log found. Run the pipeline first."}
    history = json.loads(log_path.read_text())
    return history[-1]


@mcp.tool()
def get_run_history() -> list:
    """Return metrics from all previous pipeline runs in chronological order."""
    log_path = Path(_config["mlops"]["metrics_log"])
    if not log_path.exists():
        return []
    return json.loads(log_path.read_text())


@mcp.tool()
def predict(features: dict) -> dict:
    """
    Make a regression prediction given feature values.
    Call list_features() first to see which features are required.
    Pass a dict of {feature_name: float_value} for every required feature.
    """
    missing = set(_features) - set(features)
    if missing:
        return {"error": f"Missing features: {sorted(missing)}"}

    X    = pd.DataFrame([{f: features[f] for f in _features}])
    pred = float(_model.predict(X)[0])
    return {"prediction": pred, "features_used": _features}


if __name__ == "__main__":
    mcp.run()
