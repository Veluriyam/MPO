import json

total_score = 0
count = 0
#每次计算召回行平均得分替换路径就可以
with open('/workspace/yp/MPO/logs/Qwen2.5-VL-7B/gpt-4o-mini/gpt-image/RS_DPL_Kis7/cuckoo/20260407_151428/rag_retrieval_scores.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            total_score += data['avg_score']
            count += 1

if count > 0:
    print(f"准确平均分: {total_score / count:.4f}")