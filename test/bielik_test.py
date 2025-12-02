import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# model_name = "speakleash/Bielik-1.5B-v3.0-Instruct"
# model_name = "speakleash/Bielik-7B-Instruct-v0.1"
model_name = '/home/patryk/.cache/huggingface/hub/models--speakleash--Bielik-7B-Instruct-v0.1/snapshots/ff00a0b55e9a64be87f9a8d2dcb65882f264aba4'

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto")#.to(device)
# model.enable_model_cpu_offload(device_map="auto")

messages = [
    {"role": "system", "content": "Napraw tekst wprowadzony przez użytkownika. Popraw przekręcone wyrazy. Weź pod uwagę sens całego zdania, niektóre słowa mogą być całkowicie źle, należy je zamienić innymi tak aby całość miała sens logiczny i semantyczny. Czasami słowa mogą być całkowicie pominięte, wówczas należy je dodać. Kurwa debilu jakie budynki przecież jak szczekają to wiadomo że psy."},
    {"role": "user", "content": "Ala ma kota i trzy budynek, które głośno szczekają."},
]

# 1) Użyj apply_chat_template BEZ return_tensors="pt"
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 2) Teraz dopiero tokenizujemy z paddingiem, dzięki czemu powstanie attention_mask
encoded = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True
)

input_ids = encoded.input_ids#.to(device)
attention_mask = encoded.attention_mask#.to(device)

# 3) Generowanie – musimy przekazać attention_mask
generated_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=1000,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
