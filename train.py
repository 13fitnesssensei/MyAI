from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import os
import platform
from torch.utils.data import Dataset

# Проверка железа
print(f"Платформа: {platform.platform()}")
print(f"Процессор: {platform.processor()}")
print(f"PyTorch версия: {torch.__version__}")
print(f"Доступно MPS: {torch.backends.mps.is_available()}")


# Параметры модели
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "./tinyllama-finetuned"
TRAIN_FILE = ["train.txt", "semen.txt"]  # добавил второй файл для обучения
VAL_FILE = "val.txt"

# Определение устройства
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Используемое устройство: {device.upper()}")

# Установка seed для воспроизводимости
set_seed(42)

# Оптимизация для MPS на Mac
if device == "mps":
    # Включаем оптимизации для Apple Silicon
    torch.mps.set_per_process_memory_fraction(0.8)  # Используем 80% доступной памяти
    torch.mps.empty_cache()  # Очищаем кэш перед загрузкой модели

# Загрузка токенизатора
print("\nЗагрузка токенизатора...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Кастомный датасет
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=64):  
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = load_file(file_path)
        print(f"Загружено {len(self.examples)} примеров из {os.path.basename(file_path)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze().clone()
        }

# Загрузка модели для MPS
print("\nЗагрузка модели...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if device == "mps" else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

# Настройка LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Применение LoRA к модели
model = get_peft_model(model, lora_config)

# Включаем градиенты для LoRA слоев
model.train()
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.print_trainable_parameters()

# Загрузка данных
print("\nПодготовка данных...")
try:
    # Загружаем данные из всех файлов
    all_data = []
    for file in TRAIN_FILE:
        try:
            all_data.extend(load_file(file))
        except FileNotFoundError:
            print(f"Файл {file} не найден")
            exit(1)
    
    # Сохраняем во временный файл
    temp_file = 'combined_train.txt'
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_data))
        
        # Загружаем объединенный датасет
        train_dataset = TextDataset(temp_file, tokenizer)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Загружаем валидационный датасет, если он есть
    val_dataset = TextDataset(VAL_FILE, tokenizer) if os.path.exists(VAL_FILE) else None
    if val_dataset is None:
        print("Внимание: валидационный набор данных не найден, будет использоваться только обучение")

except Exception as e:
    print(f"Ошибка при загрузке данных: {str(e)}")
    exit(1)

# Настройка коллатора
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Отключаем masked language modeling
)

# Параметры обучения

# Очистка кэша перед началом
if device == "mps":
    torch.mps.empty_cache()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,  # Уменьшаем до 1 для стабильности
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,  # Компенсируем маленький батч
    eval_strategy="no",
    save_strategy="no",
    logging_steps=10,  # Реже логируем для скорости
    learning_rate=1e-4,  # Уменьшаем скорость обучения
    weight_decay=0.01,
    fp16=False,
    optim="adamw_torch",
    max_grad_norm=1.0,
    dataloader_num_workers=0,  # Отключаем многопоточность на MPS
    remove_unused_columns=True,  # Удаляем неиспользуемые колонки
    gradient_checkpointing=False,  # Отключаем из-за проблем с градиентами
)

# Инициализация тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

# Запуск обучения
print("\nНачало обучения...")
try:
    trainer.train()
    
    # Сохранение модели
    print("\nСохранение модели...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\nОбучение успешно завершено!")
    print(f"Модель сохранена в {os.path.abspath(OUTPUT_DIR)}")
    
except Exception as e:
    print(f"\nПроизошла ошибка при обучении: {str(e)}")
    print("Попробуйте уменьшить размер батча или последовательности, если не хватает памяти.")
    