import requests
import re
import json
from bs4 import BeautifulSoup
from pathlib import Path

"""Scrapes the Retrosheet umpires page indicated by `URL` and generates a 
dict with keys Retrosheet umpire ID and values the corresponding umpire name, 
and writes the dict to file as a JSON file.
"""

if __name__ == '__main__':
    # Retrosheet umpires page for 2021
    URL = 'https://www.retrosheet.org/boxesetc/2021/YPU_2021X.htm'
    
    # output JSON name
    JSON_NAME = 'ump_id_to_ump.json'
    # output JSON path
    JSON_PATH = Path(__file__).parent / JSON_NAME

    page = requests.get(URL)
    soup = BeautifulSoup(page.text, 'html.parser')

    ump_list_html = soup.find_all('pre')[3].find_all('a')
    ump_id_dict = {re.search(r'(?<=\/\w\/P).*(?=\.)',line['href']).group(0): line.text for line in ump_list_html}

    with open(JSON_PATH, 'w') as out:
        json.dump(ump_id_dict, out)