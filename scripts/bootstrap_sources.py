"""Restore pinned upstream source required by offline audits; no model/data downloads."""
import argparse
from pathlib import Path
import subprocess
from alignlab.training.configs import COMMIT, verify_sources

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--llama-source',default='https://github.com/hiyouga/LlamaFactory.git',help='Official URL or an existing local source checkout')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
target=root/'vendor/LlamaFactory'
target.parent.mkdir(exist_ok=True)
if not target.exists():
    subprocess.run(['git','clone','--no-hardlinks','--depth','1','--branch','v0.9.3',a.llama_source,str(target)],check=True)
actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
if actual!=COMMIT: raise RuntimeError('Existing upstream checkout differs; preserved without reset')
verify_sources(root)
print('Pinned upstream source hashes verified; no weights downloaded')
