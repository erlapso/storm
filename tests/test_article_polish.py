import dspy
import unittest

from knowledge_storm.storm_wiki.modules.article_polish import StormArticlePolishingModule
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormArticle
from unittest.mock import Mock, patch

class TestStormArticlePolishingModule(unittest.TestCase):
    def setUp(self):
        # Mock the LM models
        self.mock_article_gen_lm = Mock(spec=dspy.dsp.LM)
        self.mock_article_polish_lm = Mock(spec=dspy.dsp.LM)

        # Create an instance of StormArticlePolishingModule
        self.polishing_module = StormArticlePolishingModule(
            article_gen_lm=self.mock_article_gen_lm,
            article_polish_lm=self.mock_article_polish_lm
        )

    @patch('knowledge_storm.storm_wiki.modules.article_polish.PolishPageModule')
    def test_polish_article_with_remove_duplicate(self, mock_polish_page_module):
        # Arrange
        topic = "Test Topic"
        draft_article = StormArticle(title="Test Article")
        draft_article.add_section("Introduction", "This is a test introduction.")
        draft_article.add_section("Content", "This is test content.")

        # Mock the PolishPageModule
        mock_polish_page = Mock()
        mock_polish_page_module.return_value = mock_polish_page
        mock_polish_page.return_value = dspy.Prediction(
            lead_section="This is a polished lead section.",
            page="# Content\nThis is polished content."
        )

        # Act
        polished_article = self.polishing_module.polish_article(topic, draft_article, remove_duplicate=True)

        # Assert
        self.assertIsInstance(polished_article, StormArticle)
        mock_polish_page.assert_called_once_with(
            topic=topic,
            draft_page=draft_article.to_string(),
            polish_whole_page=True
        )
        self.assertEqual(polished_article.sections[0].title, "summary")
        self.assertEqual(polished_article.sections[0].content, "This is a polished lead section.")
        self.assertEqual(polished_article.sections[1].title, "Content")
        self.assertEqual(polished_article.sections[1].content, "This is polished content.")

if __name__ == '__main__':
    unittest.main()