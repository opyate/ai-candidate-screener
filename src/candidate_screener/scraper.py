import argparse
import logging
import re
from abc import ABC, abstractmethod
from typing import List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

USER_AGENT_STRING = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

HEADERS = {
    "User-Agent": USER_AGENT_STRING,
    "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
}


class Scraper(ABC):
    def __init__(self):
        self.session = requests.Session()

    def _text_from_tags(self, tags: List) -> str:
        """Returns a string with the text from the given tags"""
        text = ""
        for t in tags:
            stripped_strings = t.stripped_strings
            for s in stripped_strings:
                text += f" {s}"
        return text

    def _remove_tags(self, soup: BeautifulSoup, tags: List) -> BeautifulSoup:
        for t in tags:
            found_tags = soup.find(t)
            if found_tags:
                if isinstance(found_tags, Tag):  # Check if found_tags is of type Tag
                    found_tags.decompose()
        return soup

    def _get_soup(self, url: str) -> BeautifulSoup:
        # Create a session to handle redirects

        # HTTP GET request
        response = self.session.get(url, timeout=20, headers=HEADERS)
        if response.status_code != 200:
            raise Exception(f"Error HTTP request: {response.status_code}")

        return BeautifulSoup(response.content, "html.parser")

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes the given text by converting it to lowercase, removing punctuation and numbers, and removing extra whitespace.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        # Convert to lowercase
        # text = text.lower()

        # Remove rows that only contain spaces
        text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)

        # Remove consecutive spaces
        text = re.sub(r" +", " ", text)

        return text

    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a"):
            path = link.get("href")
            if path and path.startswith("/"):
                path = urljoin(url, path)
            yield path

    @abstractmethod
    def scrape(self):
        pass


class TextScraper(Scraper):
    def scrape(self, url: str) -> str:
        """
        Scrapes the content of the URL and returns the text.

        Returns:
            str: The text content of the URL.
        """

        soup = self._get_soup(url)
        text = soup.get_text()
        return self._normalize_text(text)


# Write a main method to execute the scraper
class TagScraper(Scraper):
    def __init__(self):
        super().__init__()

    def scrape(self, url: str) -> str:
        """Returns the text from the titles and paragraphs of the given URL"""

        soup = self._get_soup(url)

        # Remove unwanted tags
        soup = self._remove_tags(soup, ["header", "nav", "script", "style", "footer"])

        # Find the titles and paragraphs
        title_tags = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        text_tags = soup.find_all(["article", "section", "p"])

        article = self._text_from_tags(title_tags) + self._text_from_tags(text_tags)
        return self._normalize_text(article)


class MyCustomScrapper(Scraper):
    """Your custom scrapper goes here - if you want one"""

    pass


def main(url: str):
    # Create a logger instance
    logger = logging.getLogger(__name__)

    # Create an instance of the TagScraper class
    scraper = TagScraper()

    # Call the scrape method with the sample URL
    text = scraper.scrape(url)

    # Print the returned text
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="the URL to scrape")
    args = parser.parse_args()
    url = args.url
    main(url)
