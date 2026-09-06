import sys, types, importlib.util, os
ROOT = "/Users/Chenxi/SOLab/proteina/.claude/worktrees/distogram-head"
sys.path.insert(0, ROOT)

def mkpkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    m.__package__ = name
    sys.modules[name] = m
    return m

mkpkg("proteinfoundation", os.path.join(ROOT, "proteinfoundation"))
mkpkg("proteinfoundation.nn", os.path.join(ROOT, "proteinfoundation", "nn"))
mkpkg("proteinfoundation.datasets", os.path.join(ROOT, "proteinfoundation", "datasets"))

def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m
