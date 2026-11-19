"""Heavy-dependency child process. Only invoked by explicit non-dry-run launch."""
import json
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    import torch
    from transformers import TrainerCallback
    from llamafactory.train.tuner import run_exp, export_model
    mode, config_path, run_path = sys.argv[1:4]
    run_path = Path(run_path)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    try:
        if mode == "export":
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            export_model(config)
        else:
            # Separate process discards the reference probe optimizer step and all loaded tensors.
            # Avoid initializing our CUDA context until that child has released the GPU.
            subprocess.run([sys.executable, "-m", "alignlab.training.audit", config_path,
                str(run_path / "audit")], check=True)
            torch.cuda.reset_peak_memory_stats()
            fp = sys.argv[4]
            class ProvenanceCallback(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    target = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "alignlab_fingerprint.json"
                    target.write_text(json.dumps({"fingerprint": fp, "global_step": state.global_step}), encoding="utf-8")
            run_exp(config, callbacks=[ProvenanceCallback()])
    finally:
        telemetry = {"peak_vram_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            "peak_vram_scope": "training_or_export_process_only_excludes_separate_preflight_audit"}
        (run_path / "telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
