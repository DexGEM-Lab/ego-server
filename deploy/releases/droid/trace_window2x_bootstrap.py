from __future__ import annotations
import importlib,sys
from pathlib import Path
ROOT=str(Path(__file__).resolve().parent)
if ROOT in sys.path: sys.path.remove(ROOT)
sys.path.insert(0,ROOT)
for n in tuple(sys.modules):
    if n=='ego_annotation' or n.startswith('ego_annotation.'): del sys.modules[n]
def rehome(dep,new_name):
    cls=dep.func_or_class; cls.__module__=__name__; cls.__name__=new_name; cls.__qualname__=new_name; globals()[new_name]=cls; return dep
u=importlib.import_module('ego_annotation.serving.deployment')
d=importlib.import_module('ego_annotation.serving.droid_deployment')
unidepth_app=rehome(u.UniDepthDeployment,'TraceUniDepthDeployment').bind()
droid_app=rehome(d.DroidDeployment,'TraceDroidDeployment').bind()
