from bs4 import BeautifulSoup


def parse_urls():
    with open('sitemap.xml', 'r') as file:
        xml = file.read()

    soup = BeautifulSoup(xml, 'lxml')
    locs = soup.find_all('loc')

    urls = []
    for loc in locs:
        if "pdf" not in loc.text:
            urls.append(loc.text)

    return urls
