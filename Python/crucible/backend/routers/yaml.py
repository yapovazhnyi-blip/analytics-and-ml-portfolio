from pathlib import Path
from dataclasses import dataclass
import yaml

@dataclass
class CrucibleConfig:
    db_url: str
    qdrant_host: str
    embedding_model: str
    log_level: str = "INFO"

def load_config(config_name: str = "config.yaml") -> CrucibleConfig:
    config_path = Path(__file__).resolve().parent / config_name
