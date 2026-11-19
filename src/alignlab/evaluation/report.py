"""Honest JSON, CSV and standalone HTML reports; absent measurements remain null."""
import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

from .io import write_json


def _missing(reason):
    return {"value": None, "reason": reason}


def _flatten(data, prefix=""):
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten(value, path)
        elif isinstance(value, list) and key not in ("records", "bad_cases"):
            yield path, json.dumps(value, ensure_ascii=False, allow_nan=False)
        elif not isinstance(value, list):
            yield path, value


def write_report(output_dir, *, fixture=False, preference=None, ifeval=None, engineering=None, diagnostics=None, generation_rows=None):
    out = Path(output_dir)
    # A process-test artifact may never be silently published as a real experiment.
    if not fixture and preference and preference.get("synthetic"):
        raise ValueError("synthetic preference data requires fixture=True")
    if not fixture and ifeval and ifeval.get("synthetic"):
        raise ValueError("synthetic IFEval responses require fixture=True")
    if generation_rows and any(r.get("synthetic") for r in generation_rows) and not fixture:
        raise ValueError("synthetic generations cannot be a real experiment report")
    if fixture and "fixtures" not in out.parts:
        raise ValueError("fixture reports must live under a directory named fixtures")
    if not fixture and "fixtures" in out.parts:
        raise ValueError("real reports must be separated from fixtures")
    out.mkdir(parents=True, exist_ok=True)
    title = "SYNTHETIC FIXTURE · 流程测试（不是模型成绩）" if fixture else "AlignLab 离线实验报告"
    engineering = dict(engineering or {})
    if engineering.get("ttft_seconds") is not None and engineering.get("ttft_measurement") != "streaming_first_token":
        raise ValueError("TTFT requires an actual streaming first-token measurement")
    if engineering.get("tokens_per_second") is not None and not all(engineering.get(k) is not None for k in ("hardware", "input_tokens", "output_tokens", "concurrency", "elapsed_seconds")):
        raise ValueError("throughput requires fixed hardware, lengths, concurrency and elapsed time")
    metrics = {"report_kind": "synthetic_fixture" if fixture else "experiment", "title": title,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "preference": preference if preference is not None else {"win_rate": None, "tie_score": None, "N": 0, "reason": "no human annotations"},
               "ifeval": ifeval if ifeval is not None else {"strict": {"prompt_level": None, "instruction_level": None},
                                                           "loose": {"prompt_level": None, "instruction_level": None}, "reason": "official IFEval not run"},
               "training_diagnostics": diagnostics or {"reward_accuracy": None, "reward_margin": None, "reference_revision": None, "reason": "DPO not run; diagnostic rewards are not generation win rates"},
               "engineering": {**{k: _missing("not measured") for k in ("peak_vram_bytes", "training_seconds", "tokens_per_second", "ttft_seconds")}, **engineering},
               "generation_count": len(generation_rows) if generation_rows is not None else None,
               "limitations": ["这是离线验证，不是线上业务收益。", "小样本和退化置信区间不能证明总体提升。", "未测量值保持 null；不把损失下降当能力提升。"]}
    write_json(out / "metrics.json", metrics)
    flattened = list(_flatten(metrics))
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows((key, "null" if value is None else value) for key, value in flattened)
    table = "".join(f"<tr><th>{html.escape(key)}</th><td>{html.escape('null（未测量）' if value is None else str(value))}</td></tr>" for key, value in flattened)
    bad_cases = (preference or {}).get("bad_cases", [])
    case_html = "".join("<details><summary>" + html.escape(case["case_id"] + " / " + case["prompt_id"]) + "</summary><pre>" +
                       html.escape(str(case)) + "</pre></details>" for case in bad_cases)
    (out / "report.html").write_text("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>" + html.escape(title) + "</title><style>body{font-family:system-ui;max-width:1100px;margin:32px auto;padding:16px;background:#f8fafc;color:#172033}"
        "h1{padding:24px;background:#e8eefc;border-left:6px solid #3458a8}table{border-collapse:collapse;width:100%;background:white}"
        "th,td{padding:9px;border:1px solid #dce1e9;text-align:left;overflow-wrap:anywhere}pre{white-space:pre-wrap}th{width:60%}</style>"
        "<h1>" + html.escape(title) + "</h1><p>未执行训练、官方评估或人工标注的指标为 null。流程测试不能用作简历中的模型提升成绩。</p>"
        "<table>" + table + "</table><h2>可追溯失败案例</h2>" + (case_html or "<p>尚无人工标注失败案例。</p>") + "</html>", encoding="utf-8")
    return {"status": "report_written", "kind": metrics["report_kind"], **{name: str(out / name) for name in ("summary.csv", "metrics.json", "report.html")}}
