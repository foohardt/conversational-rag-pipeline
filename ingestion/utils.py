import requests
from bs4 import BeautifulSoup


def parse_html_to_text(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        for unwanted_tag in soup(['script', 'style', 'footer']):
            unwanted_tag.decompose()
        main_content = soup.find('div', id='layout-grid__area--maincontent')

        text = main_content.get_text()
        lines = (line.strip() for line in text.splitlines())
        cleaned_text = '\n'.join(line for line in lines if line)
        return cleaned_text
    else:
        print(
            f"Error: Unable to fetch the page (Status code: {response.status_code})")
        return None


def save_text_to_file(text, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(text)
