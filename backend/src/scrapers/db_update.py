from aftenposten import Aftenposten
from nettavisen import Nettavisen
from dagbladet import Dagbladet
from nrk import NRK
from vg import VG

scrapers = [
    Aftenposten(),
    Nettavisen(),
    Dagbladet(),
    NRK(),
    VG()
]


def main():
    for scraper in scrapers:
        articles = scraper.scrape()
        scraper.save(articles)
