from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./model"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

repo_id = "Shinichii/baldur-health-ai"

model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)
