"""Fetch pinned public sources + tokenizer, then run real data governance (no weights)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from alignlab.data.sources import download_sources
from alignlab.data.pipeline import run_pipeline
from alignlab.data.tokens import load_tokenizer
from alignlab.training.configs import MODEL_REVISIONS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    p.add_argument('--sft-limit', type=int, default=20000)
    p.add_argument('--dpo-limit', type=int, default=10000)
    p.add_argument('--confirm-download', action='store_true')
    p.add_argument('--reuse-raw', help='Reuse a completed download directory and local tokenizer; no network')
    a=p.parse_args()
    if not a.confirm_download and not a.reuse_raw: p.error('Explicit --confirm-download is required')
    if not a.run_id.replace('_','').replace('-','').isalnum(): p.error('Simple run ID required')
    root=Path(__file__).resolve().parents[1]
    run=root/'runs'/'data_preparation'/a.run_id
    run.mkdir(parents=True,exist_ok=False)
    def save(name,obj): (run/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    start=time.time()
    save('code.json',{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                     'dirty':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(),
                     'command':sys.argv,'started_unix':start})
    (run/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    raw=(root/a.reuse_raw).resolve() if a.reuse_raw else root/'data/raw'/a.run_id
    processed=root/'data/processed'/a.run_id
    try:
        if a.reuse_raw:
            tokenizer_path=str(root/'models/tokenizer-qwen7b')
            if not (raw/'download_manifest.json').is_file(): raise ValueError('Completed download manifest required')
        else:
            from huggingface_hub import snapshot_download
            print('Downloading pinned 7B tokenizer only; no model weights',flush=True)
            tokenizer_path=snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision=MODEL_REVISIONS['7B'],
                local_dir=root/'models/tokenizer-qwen7b',allow_patterns=['tokenizer*','vocab.json','merges.txt','config.json'])
            print('Downloading bounded training pools and COMPLETE official evaluation exclusions',flush=True)
            download_sources(raw,allow_download=True,sft_limit=a.sft_limit,dpo_limit=a.dpo_limit,local_shards=True)
        print('Running real tokenizer cleaning / deduplication / isolation / splitting',flush=True)
        result=run_pipeline(raw/'sft.jsonl',raw/'dpo.jsonl',raw/'evaluation.jsonl',processed,
            tokenizer=load_tokenizer(tokenizer_path,MODEL_REVISIONS['7B']),max_tokens=1024,synthetic=False)
        save('status.json',{'state':'completed','seconds':time.time()-start,'raw':str(raw),'processed':str(processed),'result':result})
        print(json.dumps(result,ensure_ascii=False,default=str),flush=True)
    except BaseException as e:
        save('status.json',{'state':'failed','seconds':time.time()-start,'type':type(e).__name__,'error':str(e)})
        raise

if __name__=='__main__': main()
