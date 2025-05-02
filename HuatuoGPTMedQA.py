import json
import sys
import re

# Optional: For Elasticsearch posting
import requests

# CONFIG
ELASTIC_SEARCH_URL = None  # Example: 'http://localhost:9200/medqa/huatuo/_bulk'
SAVE_TO_FILE = True
OUTPUT_FILE = "huatuo_medqa.jsonl"

# Converts a single MedQA item to HuatuoGPT-style instruction JSON
def convert_to_huatuo_format(item):
    context = item.get('exp', '').strip()
    question = item['question'].strip()
    options = [item['opa'], item['opb'], item['opc'], item['opd']]
    correct_index = int(item['cop']) - 1
    option_letters = ['A', 'B', 'C', 'D']
    correct_answer = options[correct_index].strip()

    prompt = ""
    if context:
        prompt += f"{context}\n\n"
    prompt += f"Question: {question}\n"
    for i, opt in enumerate(options):
        prompt += f"{option_letters[i]}. {opt.strip()}\n"
    prompt += "\nPlease select the correct option (A, B, C, or D):"

    return {
        "instruction": prompt,
        "input": "",
        "output": f"{option_letters[correct_index]}. {correct_answer}"
    }

# For bulk Elasticsearch upload
def to_elasticsearch_bulk_format(json_objects):
    payload_lines = []
    for obj in json_objects:
        payload_lines.append(json.dumps({"index": {}}))
        payload_lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(payload_lines)

def main():
    huatuo_data = []

    # Read from stdin or file
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            formatted = convert_to_huatuo_format(item)
            huatuo_data.append(formatted)
        except Exception as e:
            print(f"Error parsing line: {e}", file=sys.stderr)

    # Save to file
    if SAVE_TO_FILE:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
            for obj in huatuo_data:
                fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
        print(f"Saved {len(huatuo_data)} entries to {OUTPUT_FILE}")

    # Optional: send to Elasticsearch
    if ELASTIC_SEARCH_URL:
        bulk_payload = to_elasticsearch_bulk_format(huatuo_data)
        response = requests.post(ELASTIC_SEARCH_URL, data=bulk_payload.encode('utf-8'),
                                 headers={"Content-Type": "application/x-ndjson"})
        print(f"Elasticsearch response: {response.status_code} {response.text[:200]}")

if __name__ == "__main__":
    main()
