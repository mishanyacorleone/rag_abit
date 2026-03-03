import io
import logging
from typing import List, Optional, Any

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

from app.config.config import get_settings

logger = logging.getLogger(__name__)


class PostgresManager:
    def __init__(self):
        settings = get_settings()
        self.engine = create_engine(settings.db.uri)
        self.db_name = settings.db.name

    def get_stats(self) -> dict:
        """
        Полная статистика: все таблицы, столбцы, количество строк
        """
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()

        stats = []
        with self.engine.connect() as conn:
            for table in tables:
                count = conn.execute(
                    text(f'SELECT COUNT (*) FROM "{table}"')
                ).scalar()

                columns = []
                for col in inspector.get_columns(table):
                    columns.append({
                        "table": col["name"],
                        "type": str(col["type"]),
                        "nullable": col.get("nullable", True)
                    })

                stats.append({
                    "table": table,
                    "rows": count,
                    "columns": columns
                })

        return {
            "database": self.db_name,
            "tables_count": len(stats),
            "tables": stats
        }

    def get_table_info(self, table_name) -> dict:
        """
        Информация о конкретной таблице
        """
        inspector = inspect(self.engine)

        if table_name not in inspector.get_table_names():
            raise ValueError(f"Таблица {table_name} не найдена")

        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True)
            })

        with self.engine.connect() as conn:
            count = conn.execute(
                text(f'SELECT COUNT(*) FROM "{table_name}"')
            ).scalar()

            result = conn.execute(
                text(f'SELECT * FROM "{table_name}" LIMIT 5')
            )
            column_names = list(result.keys())
            preview = [dict(zip(column_names, row)) for row in result.fetchall()]

        return {
            "table": table_name,
            "rows": count,
            "columns": columns,
            "preview": preview
        }

    def load_csv_bytes(self, file_bytes: bytes, table_name: str,
                       if_exists: str = "replace") -> dict:
        """
        Загружает csv из байтов в таблицу
        Название таблицы = название файла
        Название столбцов = название колонок в csv
        """
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8", delimiter="\t")

        if df.empty:
            raise ValueError("CSV файл пуст")

        array_columns = []
        for col in df.columns:
            if df[col].dtype == object:
                has_semicolon = df[col].dropna().str.contains(";").any()
                if has_semicolon:
                    array_columns.append(col)

        for col in array_columns:
            df[col] = df[col].apply(
                lambda x: [item.strip() for item in str(x).split(";")] if pd.notna(x) else None
            )
        
        if array_columns:
            logger.info(f"Колонки с массивами: {array_columns}")
        
        if array_columns:
            self._load_with_arrays(df, table_name, array_columns, if_exists)
        else:
            df.to_sql(
                name=table_name,
                con=self.engine,
                if_exists=if_exists,
                index=False
            )

        logger.info(
            f"CSV загружен в таблицу '{table_name}': "
            f"{len(df)} строк, столбцы: {list(df.columns)}"
        )

        return {
            "table": table_name,
            "rows_loaded": len(df),
            "columns": list(df.columns),
            "array_columns": array_columns,
            "mode": if_exists
        }
    
    def _load_with_arrays(self, df, table_name: str, array_columns: list, if_exists: str = "replace"):
        """
        Загружает DataFrame с поддержкой TEXT[] колонок
        """
        from sqlalchemy import Column, Table, MetaData, Text, ARRAY, Float, Integer
        from sqlalchemy import text as sa_text

        if if_exists == "replace":
            with self.engine.connect() as conn:
                conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
                conn.commit()
        
        columns = []
        for col in df.columns:
            if col in array_columns:
                columns.append(Column(col, ARRAY(Text)))
            else:
                sample = df[col].dropna()
                if sample.empty:
                    columns.append(Column(col, Text))
                elif pd.api.types.is_integer_dtype(sample):
                    columns.append(Column(col, Integer))
                elif pd.api.types.is_float_dtype(sample):
                    columns.append(Column(col, Float))
                else:
                    columns.append(Column(col, Text))

        metadata = MetaData()
        table = Table(table_name, metadata, *columns)
        metadata.create_all(self.engine)

        rows = []
        for _, row in df.iterrows():
            row_dict = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, list):
                    row_dict[col] = val
                elif val is None:
                    row_dict[col] = None
                elif isinstance(val, float) and pd.isna(val):
                    row_dict[col] = None
                else:
                    row_dict[col] = val
            
            rows.append(row_dict)
        
        with self.engine.connect() as conn:
            conn.execute(table.insert(), rows)
            conn.commit()


    def load_csv_file(self, file_path: str, table_name: str,
                      if_exists: str = "replace") -> dict:
        """
        Загружает CSV из файла на сервере
        """
        df = pd.read_csv(file_path, encoding="utf-8", delimiter="\t")

        if df.empty:
            raise ValueError("CSV файл пуст")

        df.to_sql(
            name=table_name,
            con=self.engine,
            if_exists=if_exists,
            index=False
        )

        logger.info(
            f"CSV '{file_path}' загружен в таблицу '{table_name}': "
            f"{len(df)} строк"
        )

        return {
            "table": table_name,
            "rows_loaded": len(df),
            "columns": list(df.columns),
            "mode": if_exists
        }

    def drop_table(self, table_name: str) -> bool:
        """
        Удаляет таблицу полностью
        """
        inspetor = inspect(self.engine)
        if table_name not in inspetor.get_table_names():
            raise ValueError(f"Таблица '{table_name}' не найдена")

        with self.engine.connect() as conn:
            conn.execute(
                text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
            )
            conn.commit()

        logger.info(f"Таблица '{table_name}' удалена")
        return True

    def clear_table(self, table_name) -> int:
        """
        Очищает таблицу
        """
        inspetor = inspect(self.engine)
        if table_name not in inspetor.get_table_names():
            raise ValueError(f"Таблица '{table_name}' не найдена")

        with self.engine.connect() as conn:
            result = conn.execute(
                text(f'DELETE FROM "{table_name}"')
            )
            conn.commit()
            deleted = result.rowcount

        logger.info(f"Таблица '{table_name}' очищена: {deleted} строк удалено")
        return deleted

    def add_row(self, table_name: str, row_data: dict) -> dict:
        """
        Добавляет одну строку в таблицу.
        row_data: {"column_name": "value", ...}
        """
        inspector = inspect(self.engine)
        if table_name not in inspector.get_table_names():
            raise ValueError(f"Таблица '{table_name}' не найдена")

        existing_columns = [col["name"] for col in inspector.get_columns(table_name)]
        for col in row_data.keys():
            if col not in existing_columns:
                raise ValueError(
                    f"Колонка '{col}' не найдена в таблице '{table_name}'"
                    f"Доступные колонки: {existing_columns}"
                )

        columns = ", ".join([f'"{col}"' for col in row_data.keys()])
        placeholders = ", ".join([f":{col}" for col in row_data.keys()])

        with self.engine.connect() as conn:
            conn.execute(
                text(f'INSERT INTO "{table_name}" ({columns}) VALUES ({placeholders})'),
                row_data
            )
            conn.commit()

        logger.info(f"Строка добавлена в '{table_name}': {row_data}")

        return {
            "table": table_name,
            "inserted": row_data
        }

    def get_tables(self) -> List[str]:
        """
        Список всех таблиц
        """
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def sql_query(self, query):
        """
        Запрос к БД из админки
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query))

            if result.returns_rows:
                return [dict(row) for row in result.mappings()]

        conn.commit()
        return {"status": "success", "rowcount": result.rowcount}
