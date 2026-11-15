from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

from src.pipeline import Config, run
from src.problem import Problem

RUNS = 10
OUT = 'results.csv'
STAGES = ['construct', 'local_search', 'lns']  # inicial, melhoria, final
FIELDS = [
    'group', 'instance', 'run', 'seed', 'requests', 'vehicles',
    *[f'{stage}_{what}' for stage in STAGES for what in ('cost', 'time_s')],
    'total_time_s', 'error',
]

def main():
    root = Path(sys.argv[1])

    # pasta > subpasta > pasta_da_instancia > arquivos: instância = pasta sem subpastas
    folders = [
        folder for folder in [root, *root.rglob('*')]
        if folder.is_dir() and not any(child.is_dir() for child in folder.iterdir())
    ]

    with open(OUT, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()

        for folder in folders:
            problem = Problem(str(folder))
            print(f'{folder.name}: {problem.requests} requests, {problem.vehicles} vehicles')

            for r in range(RUNS):
                row = {
                    'group': folder.parent.name, 'instance': folder.name, 'run': r + 1, 'seed': r,
                    'requests': problem.requests, 'vehicles': problem.vehicles,
                }
                tic = time.perf_counter()
                stats = {}

                try:
                    run(problem, Config(seed=r, log=True), stats)
                except Exception as error:  # a failed run must not lose the others
                    row['error'] = f'{type(error).__name__}: {error}'[:300]

                for stage in STAGES:
                    if f'{stage}_cost' in stats:
                        row[f'{stage}_cost'] = round(stats[f'{stage}_cost'], 4)
                        row[f'{stage}_time_s'] = round(stats[f'{stage}_time'], 3)

                row['total_time_s'] = round(time.perf_counter() - tic, 3)
                print(
                    f'  run {r + 1:2d}: construct {row.get("construct_cost", "-")} -> '
                    f'local search {row.get("local_search_cost", "-")} -> '
                    f'lns {row.get("lns_cost", "ERROR")}  ({row["total_time_s"]}s)'
                )

                writer.writerow(row)
                file.flush()


if __name__ == '__main__':
    main()