# Путь к файлу
file_path = '/Users/rustamismagilov/Desktop/Рустам/myDeepSeek/semen.txt'

# Читаем файл
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Фильтруем строки, удаляя те, что содержат LEFT-TO-RIGHT MARK (U+200E)
filtered_lines = [line for line in lines if '\u200e' not in line]

# Записываем изменения обратно в файл
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(filtered_lines)

print("Строки с LEFT-TO-RIGHT MARK (U+200E) успешно удалены.")