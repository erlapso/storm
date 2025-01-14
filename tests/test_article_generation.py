import dspy
import unittest

from knowledge_storm.collaborative_storm.modules.article_generation import ArticleGenerationModule, KnowledgeBase, KnowledgeNode
from knowledge_storm.storm_wiki.modules.article_generation import StormArticle, StormArticleGenerationModule, StormInformationTable
from unittest.mock import Mock, patch

class TestArticleGenerationModule(unittest.TestCase):
    def test_gen_section(self):
        # Mock the dspy engine
        mock_engine = Mock(spec=dspy.dsp.LM)

        # Create an instance of ArticleGenerationModule
        article_gen_module = ArticleGenerationModule(engine=mock_engine)

        # Mock KnowledgeBase and KnowledgeNode
        mock_knowledge_base = Mock(spec=KnowledgeBase)
        mock_node = Mock(spec=KnowledgeNode)

        # Set up the mock node
        mock_node.name = "Test Section"
        mock_node.content = ["Some content"]
        mock_node.synthesize_output = None
        mock_node.need_regenerate_synthesize_output = True
        mock_node.collect_all_content.return_value = {1, 2}

        # Set up the mock knowledge base
        mock_knowledge_base.info_uuid_to_info_dict = {
            1: Mock(snippets=["Snippet 1"], meta={"question": "Q1", "query": "Query1"}),
            2: Mock(snippets=["Snippet 2"], meta={"question": "Q2", "query": "Query2"})
        }

        # Mock the write_section method
        with patch.object(article_gen_module, 'write_section') as mock_write_section:
            mock_write_section.return_value = Mock(output="Generated section content")

            # Call the gen_section method
            result = article_gen_module.gen_section("Test Topic", mock_node, mock_knowledge_base)

        # Assert that the result is as expected
        self.assertEqual(result, "Generated section content")

        # Assert that the node's synthesize_output has been updated
        self.assertEqual(mock_node.synthesize_output, "Generated section content")

        # Assert that need_regenerate_synthesize_output has been set to False
        self.assertFalse(mock_node.need_regenerate_synthesize_output)

        # Assert that write_section was called with the correct arguments
        mock_write_section.assert_called_once()
        call_args = mock_write_section.call_args[1]
        self.assertEqual(call_args['topic'], "Test Topic")
        self.assertEqual(call_args['section'], "Test Section")
        self.assertIn("[1]: Snippet 1", call_args['info'])
        self.assertIn("[2]: Snippet 2", call_args['info'])

class TestStormArticleGenerationModule(unittest.TestCase):
    @patch('knowledge_storm.storm_wiki.modules.article_generation.ConvToSection')
    @patch('knowledge_storm.storm_wiki.modules.article_generation.dspy.Predict')
    def test_generate_article_single_section(self, mock_predict, mock_conv_to_section):
        # Mock the dspy LM
        mock_lm = Mock(spec=dspy.dsp.LM)

        # Create an instance of StormArticleGenerationModule
        article_gen_module = StormArticleGenerationModule(article_gen_lm=mock_lm)

        # Mock StormInformationTable
        mock_info_table = Mock(spec=StormInformationTable)
        mock_info_table.retrieve_information.return_value = [Mock(snippets=["Test snippet"])]

        # Create a StormArticle with one section
        article_with_outline = StormArticle(topic_name="Test Topic")
        article_with_outline.add_section("Test Section")

        # Mock the ConvToSection instance
        mock_section_gen = Mock()
        mock_section_gen.return_value = Mock(section="Generated section content")
        mock_conv_to_section.return_value = mock_section_gen

        # Call generate_article
        result = article_gen_module.generate_article("Test Topic", mock_info_table, article_with_outline)

        # Assert that the result is as expected
        self.assertIsInstance(result, StormArticle)
        self.assertEqual(result.topic_name, "Test Topic")
        self.assertEqual(result.get_section_content("Test Section"), "Generated section content")

        # Assert that the necessary methods were called
        mock_info_table.prepare_table_for_retrieval.assert_called_once()
        mock_info_table.retrieve_information.assert_called_once()
        mock_section_gen.assert_called_once()

        # Verify the arguments passed to the section generator
        call_args = mock_section_gen.call_args[1]
        self.assertEqual(call_args['topic'], "Test Topic")
        self.assertEqual(call_args['section'], "Test Section")
        self.assertIsInstance(call_args['collected_info'], list)