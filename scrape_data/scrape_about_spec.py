import requests
import json
from bs4 import BeautifulSoup
import os
import csv

# os.environ["HTTP_PROXY"] = "socks5://localhost:7113"
# os.environ["HTTPS_PROXY"] = "socks5://localhost:7113"


SPEC_INFO = "https://abit.magtu.ru/?bak&spec&mag"


def parse_about_spec(url=SPEC_INFO):
    response = requests.get(url).content
    soup = BeautifulSoup(response, 'lxml')

    json_specs = json.loads(soup.find('div', class_='napravleniya').find('div', class_='all_np').find('div', class_='json').text)
    
    csv_rows = []
    headers = ['code', 'profile_spec_name', 'lvl', 'link', 'FormEd', 'Plan_Budg', 'Plan_Comm', 'Years', 'Months']
    for spec in json_specs:
        if spec['lvl'] in ['bak', 'spec', "mag"]:
            code = str(spec.get('code')).lower().strip()
            title = str(spec.get('title')).lower().strip()
            lvl = str(spec.get('tag')).lower().strip()
            link = f"https://abit.magtu.ru/{spec.get('link')}".lower().strip()
            
            spec_abit_data = spec.get('abitData', [])
            for data in spec_abit_data:
                formEd = str(data.get('FormEd')).lower().strip()
                budg = str(data.get('Plan_Budg')).lower().strip()
                comm = str(data.get('Plan_Comm')).lower().strip()
                years = str(data.get('Years')).lower().strip()
                month = str(data.get('Months')).lower().strip()
    
                csv_rows.append([code, title, lvl, link, formEd, budg, comm, years, month])

    with open('result_data/spec_info.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(headers)
        writer.writerows(csv_rows)
        

        print('Данные успешно сохранены в файл')



    return True

parse_about_spec()