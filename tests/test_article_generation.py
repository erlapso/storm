import unittest

from knowledge_storm.collaborative_storm.modules.article_generation import ArticleGenerationModule
from knowledge_storm.dataclass import KnowledgeBase, KnowledgeNode
from unittest.mock import MagicMock, patch

class TestArticleGenerationModule(unittest.TestCase):
    @patch('knowledge_storm.collaborative_storm.modules.article_generation.dspy')
    def test_gen_section(self, mock_dspy):
        # Mock the engine
        mock_engine = MagicMock()

        # Create a mock KnowledgeBase
        mock_kb = MagicMock(spec=KnowledgeBase)
        mock_kb.topic = "Test Topic"
        mock_kb.info_uuid_to_info_dict = {
            1: MagicMock(snippets=["Snippet 1"], meta={"question": "Q1", "query": "Query1"}),
            2: MagicMock(snippets=["Snippet 2"], meta={"question": "Q2", "query": "Query2"})
        }

        # Create a mock KnowledgeNode
        mock_node = MagicMock(spec=KnowledgeNode)
        mock_node.name = "Test Section"
        mock_node.content = {1, 2}
        mock_node.synthesize_output = None
        mock_node.need_regenerate_synthesize_output = True
        mock_node.collect_all_content.return_value = {1, 2}

        # Mock the WriteSection predict function
        mock_write_section = MagicMock()
        mock_write_section.return_value.output = "Generated section content"

        # Create the ArticleGenerationModule instance
        module = ArticleGenerationModule(engine=mock_engine)
        module.write_section = mock_write_section

        # Mock the clean_up_section function
        with patch('knowledge_storm.collaborative_storm.modules.article_generation.clean_up_section', return_value="Cleaned section content"):
            # Call the gen_section method
            result = module.gen_section("Test Topic", mock_node, mock_kb)

        # Assertions
        self.assertEqual(result, "Cleaned section content")
        self.assertEqual(mock_node.synthesize_output, "Cleaned section content")
        self.assertFalse(mock_node.need_regenerate_synthesize_output)

        # Verify that write_section was called with correct arguments
        mock_write_section.assert_called_once()
        call_args = mock_write_section.call_args[1]
        self.assertEqual(call_args['topic'], "Test Topic")
        self.assertEqual(call_args['section'], "Test Section")
        self.assertIn("[1]: Snippet 1", call_args['info'])
        self.assertIn("[2]: Snippet 2", call_args['info'])

if __name__ == '__main__':
    unittest.main()