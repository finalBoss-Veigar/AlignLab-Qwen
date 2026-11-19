"""Small command router; data/training/evaluation implementations remain separate."""
import argparse
import json
from pathlib import Path
import sys
from .config import load_config, write_json
from .paths import project_root, resolve_path


def parser():
    p = argparse.ArgumentParser(prog="alignlab", description="AlignLab: continued SFT -> DPO, evidence before claims")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Probe actual paths, devices and available phases")
    doctor.add_argument("--output")
    sub.add_parser("smoke", help="Offline synthetic pipeline, configs and empty evaluation report; no weights")
    fixture = sub.add_parser("fixture", help="Process explicitly synthetic data")
    fixture.add_argument("--output")
    data = sub.add_parser("prepare-data", help="Process provenance-rich real JSONL using a local tokenizer")
    for name in ("sft", "dpo", "eval", "output", "tokenizer"):
        data.add_argument("--" + name, required=True)
    data.add_argument("--seed", type=int, default=42)
    data.add_argument("--max-tokens", type=int, default=1024)
    data.add_argument("--tokenizer-revision", required=True)
    data.add_argument("--minimal-filter", action="store_true")
    download = sub.add_parser("data-download", help="Explicit opt-in download of pinned data; never weights")
    download.add_argument("--output", required=True)
    download.add_argument("--confirm-download", action="store_true")
    download.add_argument("--sft-limit", type=int)
    download.add_argument("--dpo-limit", type=int)
    cfg = sub.add_parser("configs", help="Generate/check pinned LLaMA-Factory candidates")
    cfg.add_argument("--output", default="configs")
    validate = sub.add_parser("validate-config")
    validate.add_argument("path")
    validate.add_argument("--mode", choices=("train", "export"), default="train")
    for name in ("train", "export"):
        cmd = sub.add_parser(name, help="Defaults to dry-run; --execute enables real execution")
        cmd.add_argument("config")
        switches = cmd.add_mutually_exclusive_group()
        switches.add_argument("--execute", action="store_true")
        switches.add_argument("--dry-run", action="store_true")
        if name == "train":
            cmd.add_argument("--resume")
    rep = sub.add_parser("report")
    rep.add_argument("--output", required=True)
    rep.add_argument("--fixture", action="store_true")
    rep.add_argument("--preference")
    rep.add_argument("--ifeval")
    blind = sub.add_parser("blind-export")
    blind.add_argument("left")
    blind.add_argument("right")
    blind.add_argument("--output", required=True)
    blind.add_argument("--seed", type=int, default=42)
    blind.add_argument("--limit", type=int)
    score = sub.add_parser("blind-import")
    for name in ("public", "key", "annotations", "output"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--seed", type=int, default=42)
    ie = sub.add_parser("ifeval")
    for name in ("repo", "input", "responses", "output"):
        ie.add_argument("--" + name, required=True)
    ie.add_argument("--execute", action="store_true")
    srv = sub.add_parser("serve")
    srv.add_argument("--model")
    srv.add_argument("--revision")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--dry-run", action="store_true")
    client = sub.add_parser("request")
    client.add_argument("--prompt", required=True)
    client.add_argument("--url", default="http://127.0.0.1:8000")
    client.add_argument("--timeout", type=float, default=70)
    gen = sub.add_parser("generate")
    for name in ("input", "output", "model", "tokenizer", "revision", "tokenizer-revision", "size-class"):
        gen.add_argument("--" + name, required=True)
    gen.add_argument("--max-new-tokens", type=int, default=512)
    gen.add_argument("--max-input-tokens", type=int, default=2048)
    gen.add_argument("--timeout-seconds", type=float, default=1800)
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--execute", action="store_true")
    return p


def dispatch(a):
    root = project_root()
    if a.command == "doctor":
        from .tracking import environment
        result = environment()
        if a.output:
            write_json(resolve_path(a.output), result)
        return result
    if a.command == "fixture":
        from .data.pipeline import run_fixture_pipeline
        return run_fixture_pipeline(root, output_dir=resolve_path(a.output) if a.output else None)
    if a.command == "prepare-data":
        from .data.pipeline import run_pipeline
        from .data.tokens import load_tokenizer
        return run_pipeline(resolve_path(a.sft), resolve_path(a.dpo), resolve_path(a.eval), resolve_path(a.output),
            seed=a.seed, max_tokens=a.max_tokens, tokenizer=load_tokenizer(str(resolve_path(a.tokenizer)), a.tokenizer_revision),
            quality_filter=not a.minimal_filter, synthetic=False)
    if a.command == "data-download":
        from .data.sources import download_sources
        return download_sources(resolve_path(a.output), allow_download=a.confirm_download, sft_limit=a.sft_limit, dpo_limit=a.dpo_limit)
    if a.command == "configs":
        from .training.configs import generate_configs
        return generate_configs(resolve_path(a.output), project_root=root)
    if a.command == "validate-config":
        from .training.configs import validate_config
        return validate_config(load_config(resolve_path(a.path)), mode=a.mode)
    if a.command in {"train", "export"}:
        from .training.runner import run_training, run_export
        if a.command == "train":
            return run_training(resolve_path(a.config), root, dry_run=not a.execute, resume=resolve_path(a.resume) if a.resume else None)
        return run_export(resolve_path(a.config), root, dry_run=not a.execute)
    if a.command == "report":
        from .evaluation.report import write_report
        return write_report(resolve_path(a.output), fixture=a.fixture,
            preference=load_config(resolve_path(a.preference)) if a.preference else None,
            ifeval=load_config(resolve_path(a.ifeval)) if a.ifeval else None)
    if a.command == "blind-export":
        from .evaluation.blind import export_blind
        from .evaluation.io import read_jsonl
        return export_blind(read_jsonl(resolve_path(a.left)), read_jsonl(resolve_path(a.right)), resolve_path(a.output), seed=a.seed, limit=a.limit)
    if a.command == "blind-import":
        from .evaluation.blind import import_blind
        result = import_blind(resolve_path(a.public), resolve_path(a.key), resolve_path(a.annotations), seed=a.seed)
        write_json(resolve_path(a.output), result)
        return result
    if a.command == "ifeval":
        from .evaluation.ifeval import run_official
        return run_official(resolve_path(a.repo), resolve_path(a.input), resolve_path(a.responses), resolve_path(a.output), dry_run=not a.execute)
    if a.command == "serve":
        from .serving.server import serve
        return serve(resolve_path(a.model) if a.model else None, a.revision, port=a.port, dry_run=a.dry_run)
    if a.command == "request":
        from .serving.client import request
        return request(a.prompt, a.url, a.timeout)
    if a.command == "generate":
        from .evaluation.generate import generate_local
        return generate_local(resolve_path(a.model), resolve_path(a.tokenizer), resolve_path(a.input), resolve_path(a.output),
            model_revision=a.revision, tokenizer_revision=a.tokenizer_revision, size_class=a.size_class,
            max_new_tokens=a.max_new_tokens, max_input_tokens=a.max_input_tokens, device=a.device,
            timeout_seconds=a.timeout_seconds, dry_run=not a.execute)
    if a.command == "smoke":
        from .data.pipeline import run_fixture_pipeline
        from .training.configs import generate_configs
        from .training.runner import run_training
        from .evaluation.report import write_report
        from .serving.server import serve
        from .tracking import Run
        with Run("offline_smoke", {"seed": 42, "model_revision": None}, fixture=True) as run:
            data = run_fixture_pipeline(root, output_dir=run.path / "processed")
            configs = generate_configs(run.path / "configs", project_root=root)
            report = write_report(run.path / "report", fixture=True)
            service = serve(dry_run=True)
            result = {"status": "passed_offline_only", "synthetic_fixture": True, "run_dir": str(run.path),
                "data": data, "configs": configs, "report": report, "service": service,
                "training": "not_run", "real_model_metrics": None}
            write_json(run.path / "smoke_result.json", result)
            run.finish("completed", artifacts=[str(run.path / "smoke_result.json"), str(run.path / "report")])
            write_json(root / "runs" / "latest_smoke.json", {"run_dir": str(run.path)})
            return {"status": result["status"], "synthetic_fixture": True, "run_dir": str(run.path),
                    "counts": data["counts"], "configs_checked": list(configs), "report": report,
                    "training": "not_run", "full_result": str(run.path / "smoke_result.json")}
    raise ValueError("unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str, allow_nan=False))
        if isinstance(result, dict) and result.get("status") in {"failed", "error"}:
            return 1
        return 0
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as e:
        print(json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=False), file=sys.stderr)
        return 1
