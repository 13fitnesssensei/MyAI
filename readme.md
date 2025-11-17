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

