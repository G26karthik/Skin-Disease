"""Auto-update README per-class metrics table and summary KPIs.

Parses:
  runs/eval_metrics.json
  runs/throughput.json
  runs/calibration.json (optional)

Looks for markers in README.md like:
  <!-- METRIC:overall_accuracy --> ... <!-- /METRIC:overall_accuracy -->

And replaces enclosed content with formatted values.

For per-class table rows, expects markers per class field:
  <!-- CLASS:Melanocytic_nevi:precision -->...</-->

Idempotent: repeated runs will overwrite previous injected values.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path('.')
README = ROOT / 'README.md'

EVAL_PATH = Path('runs/eval_metrics.json')
THROUGHPUT_PATH = Path('runs/throughput.json')
CALIB_PATH = Path('runs/calibration.json')

def load_json(p: Path):
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}

def format_float(x, digits=4):
    return f"{x:.{digits}f}" if isinstance(x, (int,float)) else ''

def inject_marker(text: str, marker: str, value: str):
    pattern = re.compile(rf'(<!--\s*METRIC:{re.escape(marker)}\s*-->)(.*?)(<!--\s*/METRIC:{re.escape(marker)}\s*-->)', re.DOTALL)
    if not pattern.search(text):
        insertion = f"\n<!-- METRIC:{marker} -->{value}<!-- /METRIC:{marker} -->\n"
        return text + insertion
    # Use a function to avoid backreference ambiguity with numbered groups >9
    def _repl(m):
        return f"{m.group(1)}{value}{m.group(3)}"
    return pattern.sub(_repl, text)

def inject_class_metric(text: str, cls: str, metric: str, value: str):
    key = f"CLASS:{cls}:{metric}"
    pattern = re.compile(rf'(<!--\s*{re.escape(key)}\s*-->)(.*?)(<!--\s*/{re.escape(key)}\s*-->)', re.DOTALL)
    if not pattern.search(text):
        return text
    def _repl(m):
        return f"{m.group(1)}{value}{m.group(3)}"
    return pattern.sub(_repl, text)

def main():
    if not README.exists():
        print('README.md not found; aborting.')
        return
    text = README.read_text(encoding='utf-8')

    eval_metrics = load_json(EVAL_PATH)
    throughput = load_json(THROUGHPUT_PATH)
    calibration = load_json(CALIB_PATH)

    # Overall metrics
    overall = eval_metrics.get('overall', {})
    text = inject_marker(text, 'overall_accuracy', format_float(overall.get('accuracy', '')))
    text = inject_marker(text, 'overall_macro_f1', format_float(overall.get('macro_f1', '')))
    if calibration:
        text = inject_marker(text, 'ece', format_float(calibration.get('ece','')))
        text = inject_marker(text, 'brier', format_float(calibration.get('brier','')))
    if throughput:
        # Align with keys present in throughput.json
        text = inject_marker(text, 'throughput_fps', format_float(throughput.get('throughput_fps',''), digits=2))
        text = inject_marker(text, 'latency_mean', format_float(throughput.get('mean_latency_s',''), digits=5))

    # Per-class metrics
    per_class = eval_metrics.get('per_class', {})
    for cls, m in per_class.items():
        for metric in ['precision','recall','f1','support']:
            val = m.get(metric)
            if metric == 'support':
                formatted = str(val)
            else:
                formatted = format_float(val)
            text = inject_class_metric(text, cls, metric, formatted)

    README.write_text(text, encoding='utf-8')
    print('README metrics updated.')

if __name__ == '__main__':
    main()
