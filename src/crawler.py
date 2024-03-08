import logging
from time import sleep, time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from reppy.robots import Robots

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)


class Crawler:

    def __init__(self, urls=[]):
        self.visited_urls = []
        self.urls_to_visit = urls
        self.robots_cache = {}
        self.last_crawl_time = {}

    def download_url(self, url):
        return requests.get(url).text

    def get_robots_parser(self, url):
        robots_url = urljoin(url, "/robots.txt")
        if robots_url not in self.robots_cache:
            try:
                self.robots_cache[robots_url] = Robots.fetch(robots_url)
            except Exception:
                logging.exception(f"Failed to fetch robots.txt: {robots_url}")
                self.robots_cache[robots_url] = None
        return self.robots_cache[robots_url]

    def can_fetch(self, url):
        robots_parser = self.get_robots_parser(url)
        if robots_parser:
            return robots_parser.allowed(url, "*")
        return True

    def respect_crawl_delay(self, url):
        robots_parser = self.get_robots_parser(url)
        if robots_parser and robots_parser.agent("*").delay:
            crawl_delay = robots_parser.agent("*").delay
            domain = urljoin(url, "/")
            last_crawl = self.last_crawl_time.get(domain, 0)
            if time() - last_crawl < crawl_delay:
                sleep_time = crawl_delay - (time() - last_crawl)
                logging.info(
                    f"Waiting {sleep_time} seconds to respect crawl delay for {domain}"
                )
                sleep(sleep_time)
            self.last_crawl_time[domain] = time()

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a"):
            path = link.get("href")
            if path and path.startswith("/"):
                path = urljoin(url, path)
            if self.can_fetch(path):
                yield path

    def add_url_to_visit(self, url):
        if url not in self.visited_urls and url not in self.urls_to_visit:
            self.urls_to_visit.append(url)

    def crawl(self, url):
        if not self.can_fetch(url):
            logging.info(f"Not allowed to fetch: {url}")
            return
        self.respect_crawl_delay(url)
        html = self.download_url(url)
        for url in self.get_linked_urls(url, html):
            self.add_url_to_visit(url)

    def run(self):
        while self.urls_to_visit:
            url = self.urls_to_visit.pop(0)
            logging.info(f"Crawling: {url}")
            try:
                self.crawl(url)
            except Exception:
                logging.exception(f"Failed to crawl: {url}")
            finally:
                self.visited_urls.append(url)


if __name__ == "__main__":
    Crawler(urls=["https://www.example.com/"]).run()
