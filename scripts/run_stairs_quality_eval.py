"""Replay paired regular stair suites with additional contact and heading observations."""
import argparse
import json
import queue
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', type=Path)
    parser.add_argument('up_manifest', type=Path)
    parser.add_argument('down_manifest', type=Path)
    parser.add_argument('--gpus', default='0,1,2,3')
    args = parser.parse_args()
    exp = args.experiment.resolve()
    repo = Path(__file__).resolve().parents[1]
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus):
        parser.error('Duplicate GPU')
    jobs = queue.Queue()
    references = {}
    for direction, manifest in [('up', args.up_manifest), ('down', args.down_manifest)]:
        for model in json.loads(manifest.read_text())['models']:
            name = model['method'] + '_' + direction
            old = json.loads(Path(model['report']).read_text())
            config = exp / 'configs' / Path(old['args']['config']).name
            shutil.copyfile(old['args']['config'], config)
            references[name] = model['report']
            jobs.put((name, config, old))
    (exp / 'configs/references.json').write_text(json.dumps(references, indent=2))
    def worker(gpu):
        while True:
            try:
                name, config, old = jobs.get_nowait()
            except queue.Empty:
                return
            opts = old['args']
            command = [sys.executable, 'scripts/run.py', 'stairs', 'evaluate', '--gpus', gpu,
                       '--config', str(config), '--paths', str(exp / 'configs/paths.local.json'),
                       '--checkpoint', old['checkpoint'], '--output', str(exp / 'evaluation' / name),
                       '--strict-quality-trace', '--match-height-initialization', '--measure-foot-contact']
            for key in ('route','seed','speeds','heights','approaches','completion_mode','hold_seconds','duration'):
                command.extend(['--' + key.replace('_','-'), str(opts[key])])
            (exp / 'launch' / (name + '.json')).write_text(json.dumps(command, indent=2))
            with (exp / 'launch' / (name + '.log')).open('x') as log:
                subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(len(gpus)) as pool:
        list(pool.map(worker, gpus))


if __name__ == '__main__':
    main()
