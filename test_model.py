from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Базовая модель и путь к адаптеру
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./tinyllama-finetuned"

# Загрузка базовой модели и токенизатора
print("Загрузка модели и токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16,
    device_map="auto"
)

# Загрузка LoRA адаптера
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# Переводим модель в режим оценки
model.eval()

def generate_text(prompt, max_length=100, temperature=0.7):
    # Кодируем промпт и переносим на устройство модели
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Генерируем ответ
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Декодируем и возвращаем результат
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Примеры тестовых промптов
test_prompts = [
    "Меня зовут Рустам, я",
    "Моя жена",
    "Мой сын",
    "Я работаю",
    "Мой псевдоним"
]

# Запускаем тестирование
print("\nТестирование модели:")
print("=" * 50)

for prompt in test_prompts:
    print(f"\nПромпт: {prompt}")
    print("-" * 30)
    response = generate_text(prompt)
    print(f"Ответ: {response}")
    print("=" * 50)
    