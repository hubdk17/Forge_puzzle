import json

train_params = set()
with open('data/train.jsonl') as f:
    for line in f:
        p = json.loads(line)
        train_params.add(json.dumps(p['rule_params'], sort_keys=True))

novelty_params = set()
with open('data/novelty.jsonl') as f:
    for line in f:
        p = json.loads(line)
        novelty_params.add(json.dumps(p['rule_params'], sort_keys=True))

overlap = train_params & novelty_params
print(f'Train unique params: {len(train_params)}')
print(f'Novelty unique params: {len(novelty_params)}')
print(f'Overlap: {len(overlap)}')
if overlap:
    print('LEAKAGE DETECTED!')
    for o in overlap:
        print(f'  {o}')
else:
    print('NO LEAKAGE -- novelty params fully disjoint from train')

for split in ['validation', 'test']:
    sp = set()
    with open(f'data/{split}.jsonl') as f:
        for line in f:
            p = json.loads(line)
            sp.add(json.dumps(p['rule_params'], sort_keys=True))
    leak = sp - train_params
    status = 'OK' if not leak else 'FAIL: ' + str(leak)
    print(f'{split} subset of train params: {status}')

print()
print('Train params:')
for p in sorted(train_params):
    print(f'  {p}')
print('Novelty params:')
for p in sorted(novelty_params):
    print(f'  {p}')
