from bs4 import BeautifulSoup
import json
import requests
import csv
import re

PRILOZHENIA_URLS = [
    "https://www.magtu.ru/abit/17767-sroki-priema-2025-bsm.html",
    "https://www.magtu.ru/abit/17769-perechen-vi-soo-vo-2025-bs.html",
    "https://www.magtu.ru/abit/17773-perechen-vi-spo-2025-bs.html",
    "https://www.magtu.ru/abit/17775-min-max-bally-2025-bs.html",
    "https://www.magtu.ru/abit/17779-perechen-indiv-dostizh-2025-bs.html",
    "https://www.magtu.ru/abit/17781-perechen-visp-2025-m.html",
    "https://www.magtu.ru/abit/17772-perechen-indiv-dostizh-m.html",
    "https://www.magtu.ru/abit/18814-prilozhenie-14-sootvetstvie-napravlennosti-profilya-programm-bakalavriata-programm-spetsialiteta-napravlennosti-profilyu-srednego-professionalnogo-obrazovaniya.html"
]


OPTIONAL_VI_EGE = ["русский язык", "математика", "информатика", "физика",
                   "обществознание", "химия", "биология", "история", "литература",
                   "география", "иностранный язык"]


# === ПРИЛОЖЕНИЕ №1 ===

def parse_sroki_priema(url: str = PRILOZHENIA_URLS[0], output_file: str = "result_data/qdrant/sroki_priema.json"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")

    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")
        
    rows = div_table.find_all("tr")[2:]
    
    result_dict = {
        "Бакалавриат/Специалитет": {}
    }

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        if len(cols) < 3:
            continue

        stage_name, kcp_date, paid_date = cols

        result_dict["Бакалавриат/Специалитет"][stage_name] = {
            "КЦП": kcp_date,
            "Платные места": paid_date
        }

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result_dict, file, ensure_ascii=False, indent=2)

    return result_dict

# === ПРИЛОЖЕНИЕ №2 ===

def normalize_subject(subject: str) -> str:
    """
    Очистка предмета:
    - нижний регистр
    - удаление цифр
    - удаление лишних пробелов
    - замена 'математика (профиль)' на 'математика'
    """
    subject = subject.lower()
    subject = re.sub(r"\d+", "", subject)
    subject = subject.replace("(профиль)", "")
    subject = re.sub(r"\s+", " ", subject)
    return subject.strip()


def split_subjects(cell: str):
    """
    Делим предметы по '/' и нормализуем
    """
    if not cell:
        return []
    
    parts = cell.split("/")
    return [normalize_subject(p) for p in parts if p.strip()]


def parse_vi_soo_vo(url: str = PRILOZHENIA_URLS[1], output_file: str = "result_data/psql/vi_soo_vo.csv"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")
    
    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")
    
    rows = div_table.find_all("tr")[2:]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "code",
            "profile_spec_name",
            "required_vi",
            "optional_vi_ege",
            "optional_vi_vuz"
        ])

        for row in rows:
            cells = [i.get_text(strip=True) for i in row.find_all("td")]
            if len(cells) < 4:
                continue

            code, spec_name = cells[0:2]
            required_vi = [normalize_subject(i) for i in cells[2:4]]

            optional_subjects = []
            if len(cells) > 4:
                optional_subjects += split_subjects(cells[4])
            if len(cells) > 5:
                optional_subjects += split_subjects(cells[5])
            
            optional_vi_ege = []
            optional_vi_vuz = []

            for subj in optional_subjects:
                if subj in OPTIONAL_VI_EGE:
                    optional_vi_ege.append(subj)
                else:
                    optional_vi_vuz.append(subj)
            
            writer.writerow([
                code,
                spec_name,
                ";".join(required_vi),
                ";".join(optional_vi_ege),
                ";".join(optional_vi_vuz)
            ])
    return "Парсинг выполнен успешно"

# === ПРИЛОЖЕНИЕ №3 ===

def parse_vi_spo(url: str = PRILOZHENIA_URLS[2], output_file: str = "result_data/psql/vi_spo.csv"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")
    
    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")
    
    rows = div_table.find_all("tr")[2:]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "code",
            "profile_spec_name",
            "required_vi",
        ])

        for row in rows:
            cells = [i.get_text(strip=True) for i in row.find_all("td")]
            code, spec_name = cells[:2]
            required_vi = [normalize_subject(i) for i in cells[2:] if i.strip()]
            writer.writerow([
                code,
                spec_name.lower(),
                ";".join(required_vi).lower(),
            ])
    
    return "Парсинг выполнен успешно"


# === ПРИЛОЖЕНИЕ №5 ===

def parse_min_max_marks(url: str = PRILOZHENIA_URLS[3], output_file: str = "result_data/psql/min_max_marks.csv"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")

    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")
    
    rows = div_table.find_all("tr")[3:14] + div_table.find_all("tr")[15:25] + div_table.find_all("tr")[26:]
    
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "vi_name",
            "min_marks",
            "max_marks",
        ])
        for row in rows:
            cells = [i.get_text(strip=True) for i in row.find_all("td")[:3]]
            vi_name = normalize_subject(cells[0])
            min_marks, max_marks = cells[1].strip(), cells[2].strip()
            print(vi_name, min_marks, max_marks)
           
            writer.writerow([
                vi_name,
                min_marks,
                max_marks,
            ])
    
    return "Парсинг завершен успешно"


# === ПРИЛОЖЕНИЕ №9 ===

def parse_achievements(url: str = PRILOZHENIA_URLS[4], output_file: str = "result_data/qdrant/ind_achievements_bak_spec.json"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")

    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")
    
    rows = div_table.find_all("tr")[2:13]  + [div_table.find_all("tr")[14]]

    result_dict = {
        "Индивидуальные достижения": []
    }

    for row in rows:
        cols = [td.get_text(strip=True) for td in row.find_all("td")]

        achieve_name, documents, marks = cols

        result_dict["Индивидуальные достижения"].append({
            "Название индивидуального достижения": achieve_name,
            "Документы": documents,
            "Балл": marks
        })

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result_dict, file, ensure_ascii=False, indent=2)

    return result_dict
        

# === ПРИЛОЖЕНИЕ №11 ===

def parse_vi_mag(url: str = PRILOZHENIA_URLS[5], output_file: str = "result_data/psql/vi_mag.csv"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")

    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")

    rows = div_table.find_all("tr")[1:]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "code",
            "profile_spec_name",
            "vi_name",
            "min_marks",
            "max_marks",
            "vi_language",
            "vi_form"
        ])
        
        previuos_code = None
        previous_spec_name = None
        for row in rows:
            cells = [i.get_text(strip=True) for i in row.find_all("td")]
            
            if len(cells) == 9:
                code = cells[0]
                previuos_code = code
                previous_spec_name = cells[1]

                profile_spec_name = f"{cells[1]} ({cells[2]})"
                vi_name = cells[3]
                min_marks = cells[4]
                max_marks = cells[5]
                vi_lang = cells[7]
                vi_form = cells[8]
            else:
                code = previuos_code
                profile_spec_name = f"{previous_spec_name} ({cells[0]})"
                vi_name = cells[1]
                min_marks = cells[2]
                max_marks = cells[3]
                vi_lang = cells[5]
                vi_form = cells[6]

            if "или" in vi_form:
                vi_form = ["очно", "дистанционные технологии"]
            else:
                vi_form = ["очно"]

            writer.writerow([
                code,
                profile_spec_name.lower(),
                vi_name.lower(),
                min_marks,
                max_marks,
                vi_lang.lower(),
                ";".join(vi_form).lower()
            ])
    
    return "Парсинг завершен успешно"

# === ПРИЛОЖЕНИЕ №12 ХЗ КАК ПАРСИТЬ, ПОЭТОМУ ВРУЧНУЮ И ТАМ НЕМНОГО ===

# def parse_achievements_mag(url: str = PRILOZHENIA_URLS[6], output_file: str = "result_data/ind_achievements_mag.json"):
#     response = requests.get(url).content
#     soup = BeautifulSoup(response, "lxml")

#     div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")
    
#     rows = div_table.find_all("tr")[1] + div_table.find_all("tr")[3:6] + div_table.find_all("tr")[7:]

#     result_dict = {
#         "Документ об образовании с отличием": [],
#         "Научные публикации": [],
#         "Охранные документы": [],
#         "Именной сертификат ФИЭБ": []
#     }

#     for row in rows:
#         cols = [td.get_text(strip=True) for td in row.find_all("td")]

#         achieve_name, documents, marks = cols

#         result_dict["Индивидуальные достижения"].append({
#             "Название индивидуального достижения": achieve_name,
#             "Документы": documents,
#             "Балл": marks
#         })

#     with open(output_file, "w", encoding="utf-8") as file:
#         json.dump(result_dict, file, ensure_ascii=False, indent=2)

#     return result_dict


# === ПРИЛОЖЕНИЕ №14 ===

def parse_combination_vo_spo(url: str = PRILOZHENIA_URLS[7], output_file: str = "result_data/psql/vo_spo_comb.csv"):
    response = requests.get(url).content
    soup = BeautifulSoup(response, "lxml")

    div_table = soup.find("div", class_="table_wrapper").find("table").find("tbody")

    rows = div_table.find_all("tr")

    profiles = {}
    current_profile = None
  
    for row in rows:
        th = row.find("th", colspan=2)
        if th:
            current_profile = th.get_text(strip=True).lower()
            profiles[current_profile] = {
                "vo": [],
                "spo": []
            }
            continue

        tds = row.find_all("td")
        if len(tds) == 2 and current_profile:
            tds0 = tds[0].get_text(strip=True).lower()
            tds1 = tds[1].get_text(strip=True).lower()
            if tds0 == "высшее образование" or tds0 == "направление подготовки/специальность":
                continue
            
            if tds0.strip():
                profiles[current_profile]["vo"].append(tds0)
            
            if tds1.strip():
                profiles[current_profile]["spo"].append(tds1)
     
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "profile",
            "vo_spec",
            "spo_spec"
        ])

        for profile, data in profiles.items():
            vo_list = data["vo"]
            spo_list = data["spo"]
            for vo in vo_list:
                for spo in spo_list:
                    writer.writerow([
                        profile,
                        vo,
                        spo
                    ])
    return "Парсинг завершен успешно"


if __name__ == "__main__":
    parse_vi_soo_vo()
    parse_vi_spo()
    parse_min_max_marks()
    parse_vi_mag()
    parse_combination_vo_spo()

