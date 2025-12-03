# Установка Homebrew (если еще не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установка Python и pyenv
brew install pyenv
pyenv install 3.14.0
pyenv global 3.14.0

# Установка CUDA для M4 Pro (если поддерживается)
brew install cmake
brew install libomp
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu

# Настройка виртуального окружения:
python -m venv gptEnv
source gptEnv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets tqdm numpy


# запуск обучения модели
python train.py

# запуск htop в новом терминале для отслеживания загрузки устройства 
sudo htop

# запуск чата
python chat.py

# Для мониторинга обучения можно использовать TensorBoard:
tensorboard --logdir=runs

## Описание проекта

### TinyLlama Chat Assistant

Проект представляет собой чат-бота на базе модели TinyLlama 1.1B, дообученной на персональных данных. Бот использует технику RAG (Retrieval-Augmented Generation) для генерации релевантных ответов на основе предоставленной информации.

### Ключевые особенности

- **Модель**: TinyLlama-1.1B-Chat-v1.0 с дообучением
- **Технологии**:
  - PyTorch с поддержкой MPS (Apple Silicon)
  - PEFT (LoRA) для эффективной тонкой настройки
  - FAISS для векторного поиска
  - RAG для генерации ответов с учетом контекста

### Функциональность

- Обучение модели на пользовательских данных
- Интерактивный чат с контекстно-зависимыми ответами
- Поиск по базе знаний с использованием TF-IDF
- Настройка параметров генерации (температура, top-k, top-p)

### Структура проекта

- `train.py` - скрипт для обучения модели
- `chat.py` - интерактивный чат-интерфейс
- `vector_search.py` - модуль векторного поиска
- `*.txt` - текстовые файлы с данными для обучения
- `tinyllama-finetuned/` - директория с дообученной моделью

### Требования

- Python 3.14
- PyTorch
- Transformers
- PEFT
- FAISS
- scikit-learn

