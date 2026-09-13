#!/usr/bin/env python
"""Summarize train_adapter.py runs (and their posthoc_controls.py outputs) as one markdown table:
held-out NLL for base, fly adapter, direct-input control, constant-feature control, relabeled wiring,
and the fly adapter with the best test-time-training setting from the sweep."""
import argparse
import json
from pathlib import Path


def row(run):
    report = json.loads((run / 'report.json').read_text())
    manifest = json.loads((run / 'run.json').read_text())
    posthoc = json.loads((run / 'posthoc.json').read_text()) if (run / 'posthoc.json').exists() else None
    i = manifest['interface']
    block = 'FLM-equivalent (`c = 0`, K = 1)' if i['feedback'] == 0 else f"looped{' + NT signs' if manifest.get('block_source') else ', unsigned'} (`c = {i['feedback']}`, `h = {i['step_size']}`, K ≤ {i['max_iterations']})"
    backbone = Path(manifest['backbone']).name
    constant = f"{posthoc['constant_feature_adapter']['nll']:.4f}" if posthoc else '—'
    if posthoc:
        best = min(posthoc['ttt_sweep'].items(), key=lambda kv: kv[1]['nll'])
        ttt = f"{best[1]['nll']:.4f} ({best[0].split(',')[0]})"
    else:
        ttt = f"{report['fly_adapter_ttt']['nll']:.4f} (lr=0.05)"
    return (f"| {backbone} · {block} | {report['base']['nll']:.4f} | **{report['fly_adapter']['nll']:.4f}** | "
            f"{report['direct_input_adapter']['nll']:.4f} | {constant} | {report['relabeled_wiring']['nll']:.4f} | {ttt} |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+', type=Path)
    a = p.parse_args()
    print('| backbone · block | base | **fly adapter** | direct-input control | constant-feature control | relabeled wiring | fly + TTT (best lr) |')
    print('|---|---|---|---|---|---|---|')
    for folder in a.runs:
        for run in sorted(folder.glob('**/run.json')):
            print(row(run.parent))


if __name__ == '__main__':
    main()
