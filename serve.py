"""
DataForge Model & Data Research Workbench - Localhost Server
Serves the web dashboard and provides live inference API for Person A models.
"""

import os
import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import torch

# Ensure repository root is on path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from puzzle_generator.generator import generate_puzzle
from puzzle_generator.rules import RULE_REGISTRY, get_all_params
from data_utils import load_puzzles
from context_model.predict import load_model as load_ctx_model, predict_puzzle as predict_ctx
from optimization_model.predict import load_model as load_opt_model, predict_puzzle as predict_opt

DEVICE = torch.device('cpu')
CTX_MODEL = None
OPT_MODEL = None

print("Loading PyTorch model checkpoints into memory...")
try:
    CTX_MODEL, _ = load_ctx_model(str(ROOT_DIR / 'context_model' / 'checkpoint.pt'), DEVICE)
    print("  [OK] Context Model loaded.")
except Exception as e:
    print(f"  [WARN] Context Model not loaded: {e}")

try:
    OPT_MODEL, _ = load_opt_model(str(ROOT_DIR / 'optimization_model' / 'checkpoint.pt'), DEVICE)
    print("  [OK] Optimization Model loaded.")
except Exception as e:
    print(f"  [WARN] Optimization Model not loaded: {e}")


class WorkbenchHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR / 'web'), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/api/status':
            self.send_json({
                'status': 'online',
                'context_model': CTX_MODEL is not None,
                'optimization_model': OPT_MODEL is not None,
                'device': str(DEVICE),
            })
            return

        elif path == '/api/puzzles':
            split = query.get('split', ['test'])[0]
            limit = int(query.get('limit', ['20'])[0])
            rule_type = query.get('rule', [None])[0]

            file_map = {
                'train': ROOT_DIR / 'data' / 'train.jsonl',
                'validation': ROOT_DIR / 'data' / 'validation.jsonl',
                'test': ROOT_DIR / 'data' / 'test.jsonl',
                'novelty': ROOT_DIR / 'data' / 'novelty.jsonl',
            }
            p_file = file_map.get(split, file_map['test'])
            puzzles = []
            if p_file.exists():
                with open(p_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        if rule_type and item.get('rule_type') != rule_type:
                            continue
                        puzzles.append(item)
                        if len(puzzles) >= limit:
                            break
            self.send_json({'puzzles': puzzles, 'count': len(puzzles), 'split': split})
            return

        elif path == '/api/results':
            res = {}
            for name in ['sweep', 'forgetting', 'state_capacity', 'generalization']:
                fp = ROOT_DIR / 'results' / f'{name}.json'
                if fp.exists():
                    with open(fp, 'r', encoding='utf-8') as f:
                        res[name] = json.load(f)
                else:
                    res[name] = None
            self.send_json(res)
            return

        elif path == '/api/rules':
            self.send_json({
                'rules': list(RULE_REGISTRY.keys()),
                'parameters': {r: get_all_params(r) for r in RULE_REGISTRY},
            })
            return

        elif path == '/api/tech_note':
            tn_path = ROOT_DIR / 'TECHNICAL_NOTE.md'
            content = ""
            if tn_path.exists():
                with open(tn_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            self.send_json({'markdown': content})
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length).decode('utf-8')) if length > 0 else {}

        if path == '/api/generate':
            rule = body.get('rule', 'translate')
            params = body.get('params', None)
            seed = body.get('seed', int(time.time()))
            rng = np.random.default_rng(seed)
            if params is None:
                all_p = get_all_params(rule)
                params = all_p[0] if all_p else {}
            puzzle = generate_puzzle(rule_type=rule, rule_params=params, num_demos=5, rng=rng)
            self.send_json({'puzzle': puzzle})
            return

        elif path == '/api/predict/context':
            puzzle = body.get('puzzle')
            num_demos = body.get('num_demos', 5)
            if not CTX_MODEL or not puzzle:
                self.send_json({'error': 'Context model not loaded or invalid puzzle'}, status=400)
                return
            result = predict_ctx(CTX_MODEL, puzzle, num_demos=num_demos, device=DEVICE)
            self.send_json({'result': result})
            return

        elif path == '/api/predict/optimization':
            puzzle = body.get('puzzle')
            num_demos = body.get('num_demos', 5)
            K = body.get('K', 5)
            inner_lr = body.get('inner_lr', 0.05)
            if not OPT_MODEL or not puzzle:
                self.send_json({'error': 'Optimization model not loaded or invalid puzzle'}, status=400)
                return
            result = predict_opt(OPT_MODEL, puzzle, K=K, inner_lr=inner_lr, num_demos=num_demos, device=DEVICE)
            self.send_json({'result': result})
            return

        elif path == '/api/compare':
            puzzle = body.get('puzzle')
            num_demos = body.get('num_demos', 5)
            K = body.get('K', 5)
            inner_lr = body.get('inner_lr', 0.05)

            if not puzzle:
                self.send_json({'error': 'Missing puzzle'}, status=400)
                return

            ctx_res = None
            if CTX_MODEL:
                ctx_res = predict_ctx(CTX_MODEL, puzzle, num_demos=num_demos, device=DEVICE)

            opt_res = None
            if OPT_MODEL:
                opt_res = predict_opt(OPT_MODEL, puzzle, K=K, inner_lr=inner_lr, num_demos=num_demos, device=DEVICE)

            self.send_json({
                'context': ctx_res,
                'optimization': opt_res,
                'ground_truth': puzzle.get('test_pair', {}).get('output'),
                'rule_type': puzzle.get('rule_type'),
                'rule_params': puzzle.get('rule_params'),
            })
            return

        self.send_json({'error': 'Not found'}, status=404)

    def send_json(self, data, status=200):
        out = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(out)


def run_server(port=8000):
    server = HTTPServer(('127.0.0.1', port), WorkbenchHandler)
    print(f"\n=======================================================")
    print(f" DataForge Research Workbench running at:")
    print(f" >>> http://localhost:{port} <<<")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
