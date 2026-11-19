"""Create isolated 7B preflight and formal configs; never launch training."""
import argparse
from pathlib import Path
import yaml
from alignlab.training.configs import validate_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct')
    a=p.parse_args()
    if not a.run_id.replace('_','').replace('-','').isalnum(): p.error('Simple run ID required')
    root=Path(__file__).resolve().parents[1];out=root/'configs'/a.run_id
    out.mkdir(exist_ok=False)
    for scope,steps in [('preflight',5),('formal',None)]:
        prefix=f'runs/artifacts/{a.run_id}/{scope}'
        for stage in ['sft','dpo']:
            cfg=yaml.safe_load((root/f'configs/{stage}_7b.yaml').read_text(encoding='utf-8'))
            cfg['dataset_dir']=a.data
            cfg['model_name_or_path']=a.model if stage=='sft' else f'{prefix}/sft-merged'
            cfg['output_dir']=f'{prefix}/{stage}'
            if steps is not None:
                # Five updates measure memory/throughput. A full validation pass
                # here would dominate the trial and is not a model-quality claim.
                cfg.update(max_steps=steps,save_steps=steps,eval_strategy='no',logging_steps=1)
                cfg.pop('eval_steps',None)
            else:
                # One fixed training epoch, one full validation pass; no repeated
                # test-set selection or unnecessary 100-step validation sweeps.
                cfg['eval_strategy']='epoch'
                cfg.pop('eval_steps',None)
            validate_config(cfg)
            (out/f'{scope}_{stage}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
            template='export_7b.yaml' if stage=='sft' else 'export_dpo_7b.yaml'
            ex=yaml.safe_load((root/'configs'/template).read_text(encoding='utf-8'))
            ex.update(model_name_or_path=cfg['model_name_or_path'],adapter_name_or_path=cfg['output_dir'],export_dir=f'{prefix}/{stage}-merged')
            validate_config(ex,mode='export')
            (out/f'{scope}_export_{stage}.yaml').write_text(yaml.safe_dump(ex,sort_keys=False),encoding='utf-8')
    print(out)

if __name__=='__main__':main()
