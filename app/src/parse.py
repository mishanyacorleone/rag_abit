import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict, Union
import os
import re
import csv

os.environ["HTTP_PROXY"] = "socks5://localhost:7113"
os.environ["HTTPS_PROXY"] = "socks5://localhost:7113"


MAIN_LINK = "https://abit.magtu.ru/"
PER_VI_SOO_VO = "https://www.magtu.ru/abit/17769-perechen-vi-soo-vo-2025-bs.html"
PER_VI_SPO = "https://www.magtu.ru/abit/17773-perechen-vi-spo-2025-bs.html"


def parse_specs_links(url):
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'lxml')
    json_specs = json.loads(soup.find('div', class_='napravleniya').find('div', class_='all_np').find('div', class_='json').text)

    links = [f"{MAIN_LINK}{x['link']}" for x in json_specs if x['lvl'] in ['bak', 'spec', 'mag']]
    return links


def parse_about_spec(url):
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'lxml')

    profiles = soup.find('section', class_='profiles').find_all('div', class_='accordion-item')
    about_specs = []
    for profile in profiles:
        profile_name = profile.find('h2').text.strip().replace('  ', ' ')
        about_spec = profile.find('div').find('div', class_='accordion-body').find_all('ul')[-1].text.strip().replace('\n', ' ')

        context = {
            'profile_name': profile_name,
            'about_spec': about_spec
        }
        about_specs.append(context)
    return about_specs


def remove_digits(input_string: str):
    result = re.sub(r'\d+', '', input_string)

    return result


def parse_vi_soo_vo(link):
    response = requests.get(link).text
    soup = BeautifulSoup(response, 'lxml')

    table = soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[2:]
    vi_soo_vo = []
    for profile in table:
        row = profile.find_all('td')
        code = row[0].text
        profile_name = remove_digits(row[1].text).strip().replace('  ', ' ').replace(' ', '')
        vi1 = remove_digits(row[2].text)
        vi2 = remove_digits(row[3].text)
        vi3 = remove_digits(row[4].text)
        vi4 = remove_digits(row[5].text)

        if code == "07.03.03" or code == "07.03.01":
            required = f"{vi1};{vi2};{vi3};{vi4}"
            optional = "Нет"
        else:
            required = f"{vi1};{vi2}"
            split_vi3 = list(map(lambda x: x.strip(), vi3.split('/')))
            optional = ';'.join(split_vi3)

        context = {
            'code': code,
            'profile_spec_name': profile_name,
            'required_vi': required, # Обязательные ВИ
            'optional_vi': optional, # ВИ по выбору
        }

        vi_soo_vo.append(context)
    return vi_soo_vo


def parse_vi_spo(link):
    response = requests.get(link).text
    soup = BeautifulSoup(response, 'lxml')

    table = soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[2:]
    vi_spo = []
    for profile in table:
        row = profile.find_all('td')
        code = row[0].text
        profile_name = remove_digits(row[1].text).strip().replace('  ', ' ').replace(' ', '')
        vi1 = remove_digits(row[2].text)
        vi2 = remove_digits(row[3].text)
        vi3 = remove_digits(row[4].text)
        vi4 = remove_digits(row[5].text)

        if code == "07.03.03" or code == "07.03.01":
            required = f"{vi1};{vi2};{vi3};{vi4}"
        else:
            required = f"{vi1};{vi2};{vi3}"

        optional = "Нет"

        context = {
            'code': code,
            'profile_spec_name': profile_name,
            'required': required,
            'optional': optional
        }

        vi_spo.append(context)
    return vi_spo


def parse_mark_last_year(link: str, years: List[str]):
    mark_last_years = []
    for year in years:
        response = requests.get(link).text
        soup = BeautifulSoup(response, 'lxml')
        year_table = soup.find('div', id=year).find('table', class_='points').find('tbody').find_all('tr')

        for profile in year_table:
            row = profile.find_all('td')
            full_profile = row[0].text.strip().replace('  ', ' ').replace(' ', '')
            mark = row[1].text.strip()

            context = {
                'profile_spec_name': full_profile,
                'mark': mark,
                'year': year
            }

            mark_last_years.append(context)

    return mark_last_years


def parse_price(url: str = 'https://www.magtu.ru/abit/14630-stoimost-obucheniya-2022-bs.html?ckattempt=1'):
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'lxml')
    table = (soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[1:-8] +
             soup.find('div', class_='table_wrapper').find('table').find('tbody').find_all('tr')[-7:])

    prices = []
    for row in table:
        cols = row.find_all('td')
        code = cols[0].text.strip()
        profile_spec_name = cols[1].text.strip()
        education_form = cols[2].text.strip()
        price = cols[3].text.strip()

        context = {
            'code': code,
            'profile_spec_name': profile_spec_name,
            'education_form': education_form,
            'price': price
        }
        prices.append(context)
    return prices



def save_to_csv(data: List[Dict[str, Union[str, int, float]]], filename: str, encoding: str = 'utf-8'):
    if not data:
        print('Данные пусты. Файл не будет создан')
        return

    fieldnames = data[0].keys()
    with open(filename, mode='w', newline='', encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def main():
    links = parse_specs_links(MAIN_LINK)
    specs_info = []
    for link in links:
        info = parse_about_spec(link)
        for _ in info:
            specs_info.append(_)

    save_to_csv(specs_info, r'D:\mishutqa\Pycharm Projects\abitBot\data\parse\spec_info.csv')

    vi_spo = parse_vi_spo(PER_VI_SPO)
    vi_soo_vo = parse_vi_soo_vo(PER_VI_SOO_VO)
    marks_last_years = parse_mark_last_year('https://www.magtu.ru/abit/points.php', [2024, 2023])
    prices = parse_price('https://www.magtu.ru/abit/14630-stoimost-obucheniya-2022-bs.html?ckattempt=1')

    save_to_csv(vi_spo, r'D:\mishutqa\Pycharm Projects\abitBot\data\parse\vi_spo.csv')
    save_to_csv(vi_soo_vo, r'D:\mishutqa\Pycharm Projects\abitBot\data\parse\vi_soo_vo.csv')
    save_to_csv(marks_last_years, r'D:\mishutqa\Pycharm Projects\abitBot\data\parse\marks_last_years.csv')
    save_to_csv(prices, r'D:\mishutqa\Pycharm Projects\abitBot\data\parse\prices.csv')

if __name__ == "__main__":
    main()