from knowledge_storm.collaborative_storm.modules.co_storm_agents import CoStormExpert
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.interface import LMConfigs
from unittest import TestCase
from unittest.mock import MagicMock, patch

class TestCoStormExpert(TestCase):
    def test_generate_utterance(self):
        # Mock dependencies
        mock_lm_config = MagicMock(spec=LMConfigs)
        mock_runner_argument = MagicMock()
        mock_logging_wrapper = MagicMock()

        # Create a CoStormExpert instance
        expert = CoStormExpert(
            topic="AI Ethics",
            role_name="AI Ethicist",
            role_description="An expert in AI ethics and responsible AI development",
            lm_config=mock_lm_config,
            runner_argument=mock_runner_argument,
            logging_wrapper=mock_logging_wrapper
        )

        # Mock the costorm_agent_utterance_generator
        expert.costorm_agent_utterance_generator = MagicMock()
        expert.costorm_agent_utterance_generator.return_value.conversation_turn = ConversationTurn(
            role="AI Ethicist",
            raw_utterance="AI ethics is a crucial field.",
            utterance_type="Potential Answer"
        )

        # Create mock knowledge base and conversation history
        mock_knowledge_base = MagicMock(spec=KnowledgeBase)
        mock_knowledge_base.get_knowledge_base_summary.return_value = "Summary of AI ethics"
        mock_conversation_history = [
            ConversationTurn(role="User", raw_utterance="What are the main concerns in AI ethics?", utterance_type="Original Question")
        ]

        # Call the generate_utterance method
        result = expert.generate_utterance(mock_knowledge_base, mock_conversation_history)

        # Assertions
        self.assertIsInstance(result, ConversationTurn)
        self.assertEqual(result.role, "AI Ethicist")
        self.assertEqual(result.raw_utterance, "AI ethics is a crucial field.")
        self.assertEqual(result.utterance_type, "Potential Answer")

        # Verify that the necessary methods were called
        mock_knowledge_base.get_knowledge_base_summary.assert_called_once()
        expert.costorm_agent_utterance_generator.assert_called_once()
        expert.costorm_agent_utterance_generator.polish_utterance.assert_called_once()