import unittest

from knowledge_storm.collaborative_storm.modules.grounded_question_generation import GroundedQuestionGenerationModule
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.interface import Information
from unittest.mock import Mock, patch

class TestGroundedQuestionGenerationModule(unittest.TestCase):
    def setUp(self):
        self.mock_engine = Mock()
        self.module = GroundedQuestionGenerationModule(self.mock_engine)

    @patch('knowledge_storm.collaborative_storm.modules.grounded_question_generation.dspy')
    def test_forward_with_empty_unused_snippets(self, mock_dspy):
        # Arrange
        topic = "AI Ethics"
        mock_knowledge_base = Mock(spec=KnowledgeBase)
        mock_knowledge_base.get_knowledge_base_summary.return_value = "AI ethics discussion summary"
        mock_last_conv_turn = Mock(spec=ConversationTurn)
        mock_last_conv_turn.utterance = "Last utterance about AI ethics"
        unused_snippets = []

        mock_dspy.Prediction.return_value = Mock(
            raw_utterance="Raw question about AI ethics",
            utterance="Polished question about AI ethics",
            cited_info=[]
        )

        # Act
        result = self.module.forward(topic, mock_knowledge_base, mock_last_conv_turn, unused_snippets)

        # Assert
        self.assertEqual(result.raw_utterance, "Raw question about AI ethics")
        self.assertEqual(result.utterance, "Polished question about AI ethics")
        self.assertEqual(result.cited_info, [])

        mock_dspy.settings.context.assert_called_once()
        self.module.gen_focus.assert_called_once()
        self.module.polish_style.assert_called_once()

        # Verify that format_search_results was called with an empty list
        mock_dspy.collaborative_storm_utils.format_search_results.assert_called_once_with(
            [], info_max_num_words=1000
        )

if __name__ == '__main__':
    unittest.main()