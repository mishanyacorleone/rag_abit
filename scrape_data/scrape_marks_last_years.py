import requests
from bs4 import BeautifulSoup
import csv
import re

MARKS_URLS_YEARS = {
    "https://www.magtu.ru/abit/points.php": [
        "2025", "2024", "2023"
    ],
    "https://www.magtu.ru/abit/points_m.php": [
        "2025"
    ]
}

CODE_PATTERN = r"\d+\.\d+\.\d+"


def parse_marks(url_years: dict[str, list[str]], output_file: str):
    for url, years in url_years.items():
        print(f"Обрабатывается URL: {url}")
        
        response = requests.get(url).content
        soup = BeautifulSoup(response, "lxml")

        for year in years:
            print(f"Год: {year}")

            year_block = soup.find("div", id=year)
            if not year_block:
                continue

            table = year_block.find("table", class_="points")
            if not table:
                continue

            tbody = table.find("tbody")
            if not tbody:
                continue

            rows = tbody.find_all("tr")
            for row in rows:
                tds = row.find_all("td")
                if len(tds) < 2:
                    continue

                code_spec_text = tds[0].get_text(strip=True).replace(";", ",").lower()
                mark = tds[1].get_text(strip=True)

                match = re.search(CODE_PATTERN, code_spec_text)
                if not match:
                    continue

                code = match.group()
                spec_name = code_spec_text.replace(code, "").replace("    ", " ").replace("  ", " ").strip()

                with open(output_file, "a", encoding="utf-8") as file:
                    writer = csv.writer(file, delimiter="\t")
                    writer.writerow([
                        code, spec_name, mark, year
                    ])

    return "Парсинг завершен успешно"


if __name__ == "__main__":
    output_file = "result_data/marks_last_years.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "code", "profile_spec_name", "mark", "year"
        ])
    parse_marks(MARKS_URLS_YEARS, output_file)