import yaml
from pathlib import Path


def _deep_update(base: dict, new: dict) -> dict:
    """Fusion récursive : les valeurs de `new` écrasent celles de `base`."""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(path, overrides=()):
    path = Path(path)
    cfg = _load_yaml(path)

    # 1) héritage : "defaults: [base]" -> charge base.yaml situé à côté
    merged = {}
    for name in cfg.pop("defaults", []):
        merged = _deep_update(merged, _load_yaml(path.parent / f"{name}.yaml"))
    cfg = _deep_update(merged, cfg)   # le fichier enfant a priorité

    # 2) overrides CLI : training.epochs=50
    for override in overrides:
        key, val = override.split("=", 1)
        *parents, last = key.split(".")
        d = cfg
        for k in parents:
            if k not in d:
                raise KeyError(f"Override '{override}': clé '{k}' inexistante")
            d = d[k]
        if last not in d:
            raise KeyError(f"Override '{override}': clé '{last}' inexistante")
        d[last] = yaml.safe_load(val)   # gère int, float, bool, listes, null
    return cfg

