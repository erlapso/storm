import unittest

from knowledge_storm.storm_wiki.modules.storm_dataclass import Information, StormArticle
from knowledge_storm.utils import FileIOHelper
from unittest.mock import MagicMock, patch

class TestStormArticle(unittest.TestCase):
    def test_reorder_reference_index(self):
        # Create a StormArticle instance
        article = StormArticle("Test Topic")

        # Add some sections with references
        article.update_section("Introduction", "This is an introduction [1][2].")
        article.update_section("Body", "This is the body [3][1].")
        article.update_section("Conclusion", "This is the conclusion [2][3].")

        # Set up the reference dictionary
        article.reference["url_to_unified_index"] = {
            "url1": 1,
            "url2": 2,
            "url3": 3
        }
        article.reference["url_to_info"] = {
            "url1": Information(url="url1", snippets=["Snippet 1"]),
            "url2": Information(url="url2", snippets=["Snippet 2"]),
            "url3": Information(url="url3", snippets=["Snippet 3"])
        }

        # Call the method we're testing
        article.reorder_reference_index()

        # Assert that the references in the content have been updated correctly
        self.assertEqual(article.find_section(article.root, "Introduction").content,
                         "This is an introduction [1][2].")
        self.assertEqual(article.find_section(article.root, "Body").content,
                         "This is the body [3][1].")
        self.assertEqual(article.find_section(article.root, "Conclusion").content,
                         "This is the conclusion [2][3].")

        # Assert that the reference dictionary has been updated correctly
        self.assertEqual(article.reference["url_to_unified_index"], {
            "url1": 1,
            "url2": 2,
            "url3": 3
        })

        # Assert that the url_to_info dictionary remains unchanged
        self.assertEqual(len(article.reference["url_to_info"]), 3)
        self.assertIn("url1", article.reference["url_to_info"])
        self.assertIn("url2", article.reference["url_to_info"])
        self.assertIn("url3", article.reference["url_to_info"])