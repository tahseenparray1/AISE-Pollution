import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    def to_ns(d):
        for k,v in d.items():
            if isinstance(v, dict):
                d[k] = to_ns(v)
        return SimpleNamespace(**d)

    return to_ns(cfg)

