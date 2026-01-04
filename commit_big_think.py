import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np

def load_llm():
    model_id = 'speakleash/Bielik-11B-v2.6-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            # dtype=torch.bfloat16,
            trust_remote_code=True
            )

    return model, tokenizer

def big_think(model, tokenizer, prompt_path, input_text, temperature) -> str:
    prompt_file = open(prompt_path, 'r')
    sys_prompt, example = prompt_file.read().split('[EXAMPLE]')

    messages = [
        {'role': 'system', 'content': sys_prompt}
    ]

    for ex in example.split('[USER]'):
        if ex == '\n':
            continue

        usr, ass = ex.split('[ASSISTANT]')
        usr = usr.strip('\n ')
        ass = ass.strip('\n ')

        messages.append({'role': 'user', 'content': usr})
        messages.append({'role': 'system', 'content': ass})

    messages.append({'role': 'user', 'content': input_text})

    encoded = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(model.device)

    input_ids = encoded.to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(2.2*max(input_ids.shape)),
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    cleaned_output = decoded[0].split('assistant\n')[-1]

    return cleaned_output