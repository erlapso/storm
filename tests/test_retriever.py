import unittest

from knowledge_storm.storm_wiki.modules.retriever import is_valid_wikipedia_source
from urllib.parse import urlparse

class TestRetriever(unittest.TestCase):
    def test_is_valid_wikipedia_source_invalid_url(self):
        # Test with a URL from the GENERALLY_UNRELIABLE set
        invalid_url = "https://www.dailymail.co.uk/article"
        self.assertFalse(is_valid_wikipedia_source(invalid_url))

        # Test with a URL from the DEPRECATED set
        invalid_url = "https://www.rt.com/news"
        self.assertFalse(is_valid_wikipedia_source(invalid_url))

        # Test with a URL from the BLACKLISTED set
        invalid_url = "https://www.infowars.com/article"
        self.assertFalse(is_valid_wikipedia_source(invalid_url))

        # Test with a valid URL
        valid_url = "https://www.bbc.com/news/article"
        self.assertTrue(is_valid_wikipedia_source(valid_url))

if __name__ == '__main__':
    unittest.main()