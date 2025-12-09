import sqlite3
import csv

def csv_to_sqlite(csv_file: str, db_file: str, table_name: str, encoding: str = 'utf-8'):
    """
    Загружает данные из CSV в таблицу SQLite.

    :param csv_file: Путь к CSV-файлу.
    :param db_file: Путь к SQLite-файлу (например, 'university.db').
    :param table_name: Имя таблицы в базе данных.
    :param encoding: Кодировка CSV-файла.
    """
    # Открываем CSV-файл
    with open(csv_file, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = reader.fieldnames

    if not rows:
        print(f"CSV-файл {csv_file} пуст.")
        return

    # Подключаемся к SQLite (создаст файл, если его нет)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Формируем SQL-запрос для создания таблицы
    # Важно: этот способ подходит, если данные простые (строки, числа)
    # Если нужно строгое типирование — лучше указать типы вручную.
    placeholders = ', '.join(['?' for _ in headers])
    columns = ', '.join([f'"{h}" TEXT' for h in headers])  # Пока все TEXT; можно изменить по необходимости

    create_table_sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns});'
    cursor.execute(create_table_sql)

    # Вставляем данные
    insert_sql = f'INSERT INTO "{table_name}" ({", ".join([f"`{h}`" for h in headers])}) VALUES ({placeholders});'
    for row in rows:
        cursor.execute(insert_sql, [row[h] for h in headers])

    conn.commit()
    conn.close()
    print(f"Данные из {csv_file} успешно загружены в таблицу '{table_name}' базы {db_file}")

# Пример использования:
if __name__ == "__main__":
    csv_to_sqlite('parse/marks_last_years.csv', 'university.db', 'marks_last_years')
    csv_to_sqlite('parse/spec_info.csv', 'university.db', 'spec_info')
    csv_to_sqlite('parse/vi_soo_vo.csv', 'university.db', 'vi_soo_vo')
    csv_to_sqlite('parse/vi_spo.csv', 'university.db', 'vi_spo')
    csv_to_sqlite('parse/prices.csv', 'university.db', 'prices')