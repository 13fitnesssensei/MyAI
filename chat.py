from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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

# Пример простой внешней базы знаний (корпуса документах, FAQ и пр.)
DOCUMENTS = [
    "Рустам — фитнес-тренер и программист из села Завьялово.",
    "Елена Олеговна Дочия родилась 20 декабря 1984 года.",
    "Сын Рамиль родился 29 июня 2024 года.",
    "Пасынок Леон родился 17 июля 2011 года.",
    "Падчерица Артемида родилась 11 февраля 2009 года.",
    "Отец Ильдар Зинурович Исмагилов родился 12 июля 1964 года.",
    "Мать Людмила Александровна Исмагилова родилась 16 января 1965 года."
]

# Векторизация корпуса для поиска по TF-IDF
vectorizer = TfidfVectorizer().fit(DOCUMENTS)
doc_vectors = vectorizer.transform(DOCUMENTS)

def retrieve_relevant_docs(query, top_k=3):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [DOCUMENTS[i] for i in top_indices]


    
# Обновленные параметры генерации
GENERATION_CONFIG = {
    "max_new_tokens": 250,        # длина ответа
    "temperature": 0.2,           # креативность
    "top_k": 150,                # словарь выбора ответов
    "top_p": 0.8,                # случайность и разнообразие текста
    "do_sample": True,
    "repetition_penalty": 1.5,    # штраф за повторения
    "no_repeat_ngram_size": 4,    # предотвращает повторение фрагментов
    "early_stopping": True,       # останавливает генерацию, когда достигнут критерий завершения.
    "pad_token_id": tokenizer.eos_token_id,
    #  добавил
    "eos_token_id": tokenizer.eos_token_id,
    "length_penalty": 1.2,        # влияeть на длину генерации, поощряет
    "use_cache": True,

}

def clean_response(text, prompt):
    # Удаляем дублирование промпта в ответе
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    # Удаляем лишние пробелы и переносы
    return ' '.join(text.split())

print("\nМодель с RAG загружена! Введите 'выход' для завершения.")
print("=" * 50)

while True:
    try:
        prompt = input("\nВаш вопрос: ").strip()
        if prompt.lower() in ['выход', 'exit', 'quit']:
            break

        relevant_docs = retrieve_relevant_docs(prompt, top_k=3)
        retrieved_context = "\n".join(relevant_docs)

        SYSTEM_PROMPT = f"""Ты - Рустам.
Отвечай ТОЛЬКО на основе предоставленной информации.
Если чего-то не знаешь - скажи "Я не знаю" или "У меня нет такой информации".

Информация о тебе (из внешней базы знаний):
{retrieved_context}

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

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        clean_answer = clean_response(response, context)

        print(f"\nОтвет: {clean_answer}")

    except KeyboardInterrupt:
        print("\nВыход...")
        break
    except Exception as e:
        print(f"\nПроизошла ошибка: {str(e)}")
        continue

# я закомментил все что ниже :
#while True:
#    try:
#        prompt = input("\nВаш вопрос: ").strip()
#        if prompt.lower() in ['выход', 'exit', 'quit']:
#            break
#            
        
        # Улучшенный системный промпт
#        SYSTEM_PROMPT = """Ты - Рустам. 
#        Отвечай ТОЛЬКО на основе предоставленной информации.
#        Если чего-то не знаешь - скажи "Я не знаю" или "У меня нет такой информации".

#        Информация о тебе:
#        - Имя: Рустам
#        - Возраст: 32 года
#        - Профессия: фитнес-тренер и программист
#        - Псевдоним: фитнеСССенсей (fitneSSSensei)
#        - Дата рождения: 13 сентября 1991 года
#        - Место рождения: село Завьялово, Удмуртия

#        Семья:
#        - Жена: Елена Олеговна Дочия (родилась 20.12.1984)
#        - Сын: Рамиль Рустамович Исмагилов (родился 29.06.2024)
#        - Пасынок: Леон Артевич (родился 17.07.2011)
#        - Падчерица: Артемида Артевна (родилась 11.02.2009)
#        - Брат: Александр (Саня, Сашка, Брат)
#        - Сестра: Венера (Венерка, Сестра)
#        - Родители: 
#        - Отец: Ильдар Зинурович Исмагилов (родился 12.07.1964)
#        - Мать: Людмила Александровна Исмагилова (родилась 16.01.1965)

#        Отвечай КРАТКО, используя ТОЛЬКО предоставленную информацию.
#        """

#        context = f"{SYSTEM_PROMPT}\nВопрос: {prompt}\nОтвет:"

#        inputs = tokenizer(context, return_tensors="pt")
#        inputs = {k: v.to(model.device) for k, v in inputs.items()}


#        with torch.no_grad():
#            outputs = model.generate(
#                **inputs,
#                **GENERATION_CONFIG
#            )
        
        # Декодируем ответ и очищаем его
#        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#        clean_answer = clean_response(response, context)
        
#        print(f"\nОтвет: {clean_answer}")
        
#    except KeyboardInterrupt:
#        print("\nВыход...")
#        break
#    except Exception as e:
#        print(f"\nПроизошла ошибка: {str(e)}")
#        continue
    