import requests
from bs4 import BeautifulSoup
import csv

PRICES_URLS = [
    "https://www.magtu.ru/abit/14630-stoimost-obucheniya-2022-bs.html",
    "https://www.magtu.ru/abit/14929-stoimost-obucheniya-2022-m.html"
]


def parse_combination_vo_spo(url: str, output_file: str):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")

    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")

    rows = div_table.find_all("tr")[1:]
    
    with open(output_file, "a", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for row in rows:
            temp_td = row.find("td", colspan=4)
            if temp_td:
                continue
            

            tds = [i.get_text().lower() for i in row.find_all("td")]
            code, spec_name, ed_form, price = tds
            writer.writerow([
                code,
                spec_name,
                ed_form,
                price
            ])
    return "Парсинг завершен успешно"


if __name__ == "__main__":
    output_file = "result_data/psql/prices.csv"
    with open(output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "code",
            "spec_name",
            "ed_form",
            "price"
        ])
    for url in PRICES_URLS:
        parse_combination_vo_spo(url, output_file)