from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./tinyllama-finetuned"

print("Загрузка модели...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    print("Модель успешно загружена на устройство:", next(model.parameters()).device)
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    
# Обновленные параметры генерации
GENERATION_CONFIG = {
    "max_new_tokens": 7,          # Уменьшаем длину ответа
    "temperature": 0.1,           # Понижаем температуру для более предсказуемых ответов
    "top_k": 20,                  # Уменьшаем словарь
    "top_p": 0.9,                 # Немного уменьшаем
    "do_sample": True,
    "repetition_penalty": 1.5,    # Увеличиваем штраф за повторения
    "no_repeat_ngram_size": 2,    # Уменьшаем размер запрещенных N-грамм
    "pad_token_id": tokenizer.eos_token_id
}

def clean_response(text, prompt):
    # Удаляем дублирование промпта в ответе
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    # Удаляем лишние пробелы и переносы
    return ' '.join(text.split())

print("\nМодель загружена! Введите 'выход' для завершения.")
print("=" * 50)

while True:
    try:
        prompt = input("\nВаш вопрос: ").strip()
        if prompt.lower() in ['выход', 'exit', 'quit']:
            break
            
        
        # Улучшенный системный промпт
        SYSTEM_PROMPT = """Ты - Рустам. 
        Отвечай ТОЛЬКО на основе предоставленной информации.
        Если чего-то не знаешь - скажи "Я не знаю" или "У меня нет такой информации".

        Информация о тебе:
        - Имя: Рустам
        - Возраст: 32 года
        - Профессия: фитнес-тренер и программист
        - Псевдоним: фитнеСССенсей (fitneSSSensei)
        - Дата рождения: 13 сентября 1991 года
        - Место рождения: село Завьялово, Удмуртия

        Семья:
        - Жена: Елена Олеговна Дочия (родилась 20.12.1984)
        - Сын: Рамиль Рустамович Исмагилов (родился 29.06.2024)
        - Пасынок: Леон Артевич (родился 17.07.2011)
        - Падчерица: Артемида Артевна (родилась 11.02.2009)
        - Брат: Александр (Саня, Сашка, Брат)
        - Сестра: Венера (Венерка, Сестра)
        - Родители: 
        - Отец: Ильдар Зинурович Исмагилов (родился 12.07.1964)
        - Мать: Людмила Александровна Исмагилова (родилась 16.01.1965)

        Отвечай КРАТКО, используя ТОЛЬКО предоставленную информацию.
        """

        context = f"{SYSTEM_PROMPT}\nВопрос: {prompt}\nОтвет:"

        inputs = tokenizer(context, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}


        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG
            )
        
        # Декодируем ответ и очищаем его
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_answer = clean_response(response, context)
        
        print(f"\nОтвет: {clean_answer}")
        
    except KeyboardInterrupt:
        print("\nВыход...")
        break
    except Exception as e:
        print(f"\nПроизошла ошибка: {str(e)}")
        continue
    