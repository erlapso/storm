import unittest

from knowledge_storm.storm_wiki.modules.outline_generation import StormArticle, StormInformationTable, StormOutlineGenerationModule
from unittest.mock import MagicMock, patch

class TestStormOutlineGenerationModule(unittest.TestCase):
    @patch('knowledge_storm.storm_wiki.modules.outline_generation.dspy')
    def test_generate_outline_with_old_outline(self, mock_dspy):
        # Create a mock LM
        mock_lm = MagicMock()
        mock_dspy.dsp.LM = MagicMock(return_value=mock_lm)

        # Create an instance of StormOutlineGenerationModule
        outline_gen = StormOutlineGenerationModule(mock_lm)

        # Create test data
        topic = "Test Topic"
        information_table = StormInformationTable(conversations=[])
        old_outline = StormArticle(topic="Test Topic", outline="# Old Section\n## Old Subsection")

        # Mock the WriteOutline.forward method
        mock_write_outline = MagicMock()
        mock_write_outline.return_value.outline = "# New Section\n## New Subsection"
        outline_gen.write_outline = mock_write_outline

        # Call generate_outline with an old_outline
        result = outline_gen.generate_outline(topic, information_table, old_outline)

        # Assert that the result is a StormArticle
        self.assertIsInstance(result, StormArticle)

        # Assert that the outline in the result matches the expected new outline
        self.assertEqual(result.outline, "# New Section\n## New Subsection")

        # Verify that write_outline was called with the correct arguments
        mock_write_outline.assert_called_once()
        call_args = mock_write_outline.call_args[1]
        self.assertEqual(call_args['topic'], topic)
        self.assertEqual(call_args['dlg_history'], [])
        self.assertIsNone(call_args['old_outline'])  # Old outline is not passed to write_outline

if __name__ == '__main__':
    unittest.main()