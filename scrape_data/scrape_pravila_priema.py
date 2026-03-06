import re
import json
import pdfplumber


def parse_structure(file_path: str) -> dict:
    chapter_pattern = re.compile(r'^[IVXLCDM]+\.\s+.*')

    item_pattern = re.compile(r'^\d+\.\s+.*')

    result_dict = {}

    chapter_title_lines = []
    current_item_lines = []

    current_chapter_key = None

    is_collecting_chapter = False

    with pdfplumber.open(file_path) as pdf:
        pages_to_parse = pdf.pages[1:39]

        for page in pages_to_parse:
            text = page.extract_text()
            if not text:
                continue

            lines = text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if chapter_pattern.match(line):
                    if current_chapter_key and current_item_lines:
                        result_dict[current_chapter_key].append(" ".join(current_item_lines))

                    current_item_lines = []
                    chapter_title_lines = [line]

                    is_collecting_chapter = True
                    current_chapter_key = None
                
                elif item_pattern.match(line):
                    if is_collecting_chapter:
                        current_chapter_key = " ".join(chapter_title_lines)
                        result_dict[current_chapter_key] = []
                        is_collecting_chapter = False
                    
                    if current_chapter_key and current_item_lines:
                        result_dict[current_chapter_key].append(" ".join(current_item_lines))
                    
                    current_item_lines = [line]
                
                else:
                    if is_collecting_chapter:
                        chapter_title_lines.append(line)
                    elif current_chapter_key is not None:
                        current_item_lines.append(line)

        if current_chapter_key and current_item_lines:
            result_dict[current_chapter_key].append(" ".join(current_item_lines))

    return result_dict

file_name = "initial_data/pravila_priema.pdf"

try:
    parsed_data = parse_structure(file_name)

    with open("result_data/qdrant/pravila_priema.json", "w", encoding="utf-8") as file:
        json.dump(parsed_data, file, ensure_ascii=False, indent=2)

    for chapter, items in parsed_data.items():
        print(f"ГЛАВА: {chapter}")
        for item in items:
            print(f" --- Пункт (длина {len(item)}): {item[:100]}...")
        print("-" * 40)
    
except FileNotFoundError:
    print(f"Файл не найден")
except Exception as e:
    print(f"Произошла ошибка: {e}")