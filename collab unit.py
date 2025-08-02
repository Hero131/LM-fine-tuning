%%capture
!pip uninstall -y torch torchvision torchaudio
!pip install torch==2.1.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install --force-reinstall unsloth vllm==0.8.5.post1 synthetic-data-kit

import unsloth
from unsloth.dataprep import SyntheticDataKit

generator = SyntheticDataKit.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
)

generator.prepare_qa_generation(
    output_folder="data",
    temperature=0.7,
    top_p=0.95,
    overlap=64,
    max_generation_tokens=512,
)

!synthetic-data-kit system-check

!synthetic-data-kit -c synthetic_data_kit_config.yaml ingest "https://arxiv.org/html/2412.09871v1"

filenames = generator.chunk_data("data/output/arxiv_org.txt")

import time
for filename in filenames[:3]:
    !synthetic-data-kit -c synthetic_data_kit_config.yaml create {filename} --num-pairs 25 --type "qa"
    time.sleep(2)

qa_pairs_filenames = [f"data/generated/arxiv_org_{i}_qa_pairs.json" for i in range(len(filenames[:3]))]

for filename in qa_pairs_filenames:
    !synthetic-data-kit -c synthetic_data_kit_config.yaml save-as {filename} -f ft

from datasets import Dataset
import pandas as pd

final_filenames = [f"data/final/arxiv_org_{i}_qa_pairs_ft.json" for i in range(len(filenames[:3]))]

conversations = pd.concat([pd.read_json(name) for name in final_filenames]).reset_index(drop=True)
dataset = Dataset.from_pandas(conversations)

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return { "text": texts }

dataset = dataset.map(formatting_prompts_func, batched=True)

from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

messages = [{"role": "user", "content": "What is the Byte Latent Transformer?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=256, temperature=0.1)

messages = [{"role": "user", "content": "What are some benefits of the BLT?"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=256, temperature=0.1)

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
