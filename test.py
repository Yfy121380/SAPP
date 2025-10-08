import json

with open('./p_dict/eps_1.0_top_20.txt', 'r', encoding='utf-8') as f1:
    candidate_data = json.load(f1)

with open('./sim_word_dict/eps_1.0_top_20.txt', 'r', encoding='utf-8') as f2:
    probability_data = json.load(f2)

result = {}

for key in candidate_data:
    if key in probability_data:
        candidates = candidate_data[key]
        probabilities = probability_data[key]

        combined = list(zip(candidates, probabilities))

        combined.sort(key=lambda x: x[1], reverse=True)

        top_10 = [item[1] for item in combined[:10]]
        result[key] = top_10

with open('./sim_word_dict/top_10.txt', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("处理完成")
