# Путь к файлу
file_path = '/Users/rustamismagilov/Desktop/Рустам/myDeepSeek/kirillVit.txt'

# Читаем файл
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Удаляем временные метки формата [дд/мм/гг, чч:мм:сс]
import re
filtered_lines = []
for line in lines:
    # Удаляем временные метки в начале строки
    cleaned_line = re.sub(r'^\[\d{1,2}/\d{1,2}/\d{2,4}, \d{2}:\d{2}:\d{2}\] ', '', line)
    filtered_lines.append(cleaned_line)

# Записываем изменения обратно в файл
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(filtered_lines)

print("Временные метки успешно удалены из файла.")