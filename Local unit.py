import os
import re
import json
import requests
from datetime import datetime

file_path = '\Users\pauls\OneDrive\Desktop\llm training (research paper)/1mb/case_1.txt'
base_filename = os.path.splitext(os.path.basename(file_path))[0]

with open(file_path, 'r', encoding='utf-8') as f:
    full_text = f.read()
print(f"done"
      f" Loaded {base_filename}.txt")

def split_text(text, max_chunk_size=1000):
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

chunks = split_text(full_text)
print(f"Split into {len(chunks)} chunks.")

def ask_ollama_api(prompt, model="llama3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    result = response.json()
    return result.get("response", "").strip()

text_output_dir = "outputs_txt"
json_output_dir = "outputs_json"
os.makedirs(text_output_dir, exist_ok=True)
os.makedirs(json_output_dir, exist_ok=True)

text_output_file = os.path.join(text_output_dir, f"{base_filename}_qa.txt")
json_output_file = os.path.join(json_output_dir, f"{base_filename}_qa.json")

qa_text_results = []
with open(text_output_file, "w", encoding='utf-8') as f:
    for i, chunk in enumerate(chunks):
        prompt = f"Generate 3 Q&A pairs from the following legal text:\n\n{chunk}"
        print(f" Processing chunk {i+1}/{len(chunks)}...")
        response = ask_ollama_api(prompt)
        f.write(f"--- Chunk {i+1} ---\n{response}\n\n")
        qa_text_results.append(response)

print(f" All Q&A pairs saved to {text_output_file}")

qa_pairs = []
for response in qa_text_results:
    matches = re.findall(r'Q:\s*(.?)\nA:\s(.*?)(?:\n\n|$)', response, re.DOTALL)
    for q, a in matches:
        qa_pairs.append({"question": q.strip(), "answer": a.strip()})

with open(json_output_file, 'w', encoding='utf-8') as jf:
    json.dump(qa_pairs, jf, indent=4, ensure_ascii=False)

print(f" Converted {len(qa_pairs)} Q&A pairs to JSON at {json_output_file}")
