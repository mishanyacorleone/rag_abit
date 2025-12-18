import sqlite3

# === Настройки ===
DB_PATH = "university.db"  # Замени на путь к твоей SQLite-базе
TABLE_NAME = "vi_soo_vo"  # Измени на "vi_spo", если нужно обработать другую таблицу

# === Список предметов ЕГЭ (в нижнем регистре) ===
EGE_SUBJECTS = {
    "русский язык", "математика", "физика", "химия", "история",
    "обществознание", "информатика", "биология", "география",
    "иностранный язык", "литература"
}

def drop_and_recreate_table(conn, table_name):
    cursor = conn.cursor()

    # Получаем текущую схему таблицы
    cursor.execute(f"PRAGMA table_info(`{table_name}`)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]

    # Удаляем старые колонки из списка, если они есть
    if "optional_vi_ege" in column_names:
        column_names.remove("optional_vi_ege")
    if "optional_vi_vuz" in column_names:
        column_names.remove("optional_vi_vuz")

    # Убираем optional_vi — его мы удаляем навсегда
    include_optional_vi = "optional_vi" in column_names
    if include_optional_vi:
        column_names.remove("optional_vi")

    # Новые колонки (в конец)
    new_columns = column_names + ["optional_vi_ege", "optional_vi_vuz"]

    # Формируем список колонок со старыми типами (предполагаем TEXT)
    column_defs = []
    for col in new_columns:
        if col in ("optional_vi_ege", "optional_vi_vuz"):
            column_defs.append(f'"{col}" TEXT')
        else:
            # Найдём тип оригинальной колонки
            orig_col = next((c for c in columns if c[1] == col), None)
            col_type = orig_col[2] if orig_col else "TEXT"
            column_defs.append(f'"{col}" {col_type}')

    # Создаём новую таблицу
    new_table = f"{table_name}_new"
    create_sql = f'CREATE TABLE "{new_table}" ({", ".join(column_defs)})'
    cursor.execute(create_sql)

    # Переносим данные
    select_cols = [f'"{col}"' for col in column_names]
    if include_optional_vi:
        select_cols.append('"optional_vi"')

    cursor.execute(f'SELECT {", ".join(select_cols)} FROM "{table_name}"')
    rows = cursor.fetchall()

    insert_cols = [f'"{col}"' for col in new_columns]
    placeholders = ", ".join(["?"] * len(insert_cols))

    for row in rows:
        # Преобразуем optional_vi, если он был
        optional_vi = None
        if include_optional_vi:
            optional_vi = row[-1]  # последний элемент — optional_vi

        # Разделяем optional_vi
        ege_items = []
        vuz_items = []

        if optional_vi and isinstance(optional_vi, str):
            # Разбиваем строго по ";", без strip (как в исходнике)
            items = [item for item in optional_vi.split(";") if item]  # удаляем пустые
            for item in items:
                item_lower = item.lower()
                if any(ege_subj in item_lower for ege_subj in EGE_SUBJECTS):
                    ege_items.append(item)
                else:
                    vuz_items.append(item)

        # Собираем без пробелов: "предмет1;предмет2"
        ege_str = ";".join(ege_items)
        vuz_str = ";".join(vuz_items)

        # Формируем новую строку
        new_row = list(row[:-1]) if include_optional_vi else list(row)
        new_row.extend([ege_str, vuz_str])

        cursor.execute(f'INSERT INTO "{new_table}" VALUES ({placeholders})', new_row)

    # Заменяем старую таблицу на новую
    cursor.execute(f'DROP TABLE "{table_name}"')
    cursor.execute(f'ALTER TABLE "{new_table}" RENAME TO "{table_name}"')

    conn.commit()
    print(f"✅ Таблица '{table_name}' обновлена: optional_vi удалён, добавлены ege/vuz без пробелов.")

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    result = cursor.execute("SELECT * FROM vi_soo_vo")
    for res in result:
        print(res)
    # try:
    #     drop_and_recreate_table(conn, TABLE_NAME)
    # except Exception as e:
    #     print(f"❌ Ошибка: {e}")
    #     conn.rollback()
    # finally:
    #     conn.close()

if __name__ == "__main__":
    main()