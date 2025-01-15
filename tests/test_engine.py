import unittest

from knowledge_storm.collaborative_storm.engine import CoStormRunner, CollaborativeStormLMConfigs, RunnerArgument
from knowledge_storm.logging_wrapper import LoggingWrapper
from unittest.mock import MagicMock, patch

class TestCoStormRunner(unittest.TestCase):
    @patch('knowledge_storm.collaborative_storm.engine.BingSearch')
    def test_warm_start_rag_only_baseline_mode(self, mock_bing_search):
        # Mock the dependencies
        mock_lm_config = MagicMock(spec=CollaborativeStormLMConfigs)
        mock_runner_argument = MagicMock(spec=RunnerArgument)
        mock_runner_argument.rag_only_baseline_mode = True
        mock_runner_argument.topic = "Test Topic"
        mock_logging_wrapper = MagicMock(spec=LoggingWrapper)

        # Create the CoStormRunner instance
        runner = CoStormRunner(
            lm_config=mock_lm_config,
            runner_argument=mock_runner_argument,
            logging_wrapper=mock_logging_wrapper
        )

        # Mock the pure_rag_agent
        mock_pure_rag_agent = MagicMock()
        mock_pure_rag_agent.generate_topic_background.return_value = MagicMock()
        runner.discourse_manager.pure_rag_agent = mock_pure_rag_agent

        # Call the warm_start method
        runner.warm_start()

        # Assertions
        self.assertIsNotNone(runner.knowledge_base)
        self.assertEqual(runner.knowledge_base.topic, "Test Topic")
        self.assertEqual(len(runner.conversation_history), 1)
        mock_pure_rag_agent.generate_topic_background.assert_called_once()
        runner.knowledge_base.update_from_conv_turn.assert_called_once()
        self.assertTrue(runner.knowledge_base.update_from_conv_turn.call_args[1]['insert_under_root'])