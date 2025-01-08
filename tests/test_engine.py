import unittest

from knowledge_storm.collaborative_storm.engine import CoStormRunner, CollaborativeStormLMConfigs, RunnerArgument
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.logging_wrapper import LoggingWrapper
from unittest.mock import MagicMock, patch

class TestCoStormRunner(unittest.TestCase):

    @patch('knowledge_storm.collaborative_storm.engine.BingSearch')
    @patch('knowledge_storm.collaborative_storm.engine.os.getenv')
    def test_warm_start_rag_only_baseline_mode(self, mock_getenv, mock_bing_search):
        # Mock environment variables
        mock_getenv.return_value = 'openai'

        # Create a mock for LoggingWrapper
        mock_logging_wrapper = MagicMock(spec=LoggingWrapper)

        # Initialize CollaborativeStormLMConfigs
        lm_config = CollaborativeStormLMConfigs()
        lm_config.init(lm_type='openai')

        # Create RunnerArgument with rag_only_baseline_mode set to True
        runner_argument = RunnerArgument(
            topic="Test Topic",
            rag_only_baseline_mode=True
        )

        # Initialize CoStormRunner
        runner = CoStormRunner(
            lm_config=lm_config,
            runner_argument=runner_argument,
            logging_wrapper=mock_logging_wrapper
        )

        # Mock the generate_topic_background method of PureRAGAgent
        mock_conv_turn = ConversationTurn(
            role="PureRAG",
            raw_utterance="This is a background on Test Topic.",
            utterance_type="Background"
        )
        runner.discourse_manager.pure_rag_agent.generate_topic_background = MagicMock(return_value=mock_conv_turn)

        # Call warm_start method
        runner.warm_start()

        # Assertions
        self.assertIsNotNone(runner.knowledge_base)
        self.assertIsInstance(runner.knowledge_base, KnowledgeBase)
        self.assertEqual(runner.knowledge_base.topic, "Test Topic")

        self.assertEqual(len(runner.conversation_history), 1)
        self.assertEqual(runner.conversation_history[0], mock_conv_turn)

        # Verify that generate_topic_background was called
        runner.discourse_manager.pure_rag_agent.generate_topic_background.assert_called_once()

        # Verify that update_from_conv_turn was called with the correct arguments
        runner.knowledge_base.update_from_conv_turn.assert_called_once_with(
            conv_turn=mock_conv_turn,
            allow_create_new_node=True,
            insert_under_root=True
        )