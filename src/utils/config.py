import json
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def prepare_experiment_dirs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(cfg.get("output_dir", "results/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(output_dir)

    log_cfg = cfg.get("logging", {})
    default_log_dir = output_dir / "logs"
    log_dir = Path(log_cfg.get("log_dir", default_log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_cfg["log_dir"] = str(log_dir)
    cfg["logging"] = log_cfg

    return cfg
