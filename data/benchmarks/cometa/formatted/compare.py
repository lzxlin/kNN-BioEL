from tqdm import tqdm


def load_data(path):
    data = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data.add(line.strip())
    return data


data1 = load_data('test_dictionary.txt')
data2 = load_data('test2_dictionary.txt')

print(len(data1), len(data2))

for line in tqdm(data1):
    if line not in data2:
        print(line)
