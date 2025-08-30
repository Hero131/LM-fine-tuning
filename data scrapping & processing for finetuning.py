import os
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
os.environ["TRANSFORMERS_VERBOSITY"] = "info"

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
qna_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def generate_qna(text):
    prompt = (
        "You are an expert in Indian law and judicial systems.\n\n"
        "Given the following legal content, extract 10 relevant question-answer pairs. "
        "Each question must be meaningful and each answer must be based only on the content provided.\n\n"
        "Format strictly as:\n"
        "question: <question>\n"
        "answer: <answer>\n\n"
        "Legal Content:\n"
        f"{text.strip()}\n\n"
        "Extracted Q&A pairs:"
    )
    result = qna_pipe(prompt, max_new_tokens=512, do_sample=False, temperature=0.3)
    return result[0]['generated_text'].strip()

input_path = "/content/momo/experiment.txt" 
output_path = "qna_output.txt"

with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
chunk_size = 1500
chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
with open(output_path, "w", encoding="utf-8") as out_file:
    for i, chunk in tqdm(enumerate(chunks, 1), total=len(chunks)):
        qna_text = generate_qna(chunk)
        out_file.write(f"===== case_{i}.txt - Chunk {i} =====\n")
        out_file.write(qna_text + "\n\n")
        
print("All Q&A pairs saved to:", output_path)
