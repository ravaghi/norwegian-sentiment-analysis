from scrapers.aftenposten import Aftenposten
from scrapers.nettavisen import Nettavisen
from scrapers.dagbladet import Dagbladet
from scrapers.nrk import NRK
from scrapers.vg import VG

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
