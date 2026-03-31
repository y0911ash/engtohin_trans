import json

with open('ATML_Lab6_Solution.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and '# Dataloaders' in "".join(cell['source']):
        new_source = []
        found_train = False
        for line in cell['source']:
            if 'valid_data = data_process' in line and not found_train:
                new_source.append("train_data = data_process(dataset['train'])\n")
                found_train = True
            new_source.append(line)
        cell['source'] = new_source

with open('ATML_Lab6_Solution.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)
