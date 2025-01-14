import unittest

from knowledge_storm.collaborative_storm.engine import CoStormRunner, CollaborativeStormLMConfigs, RunnerArgument
from knowledge_storm.dataclass import ConversationTurn, KnowledgeBase
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.storm_wiki.engine import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormArticle, StormInformationTable
from unittest.mock import MagicMock, patch

class TestCoStormRunner(unittest.TestCase):
    @patch('knowledge_storm.collaborative_storm.engine.BingSearch')
    def test_warm_start_rag_only_baseline_mode(self, mock_bing_search):
        # Set up mock objects
        mock_lm_config = MagicMock(spec=CollaborativeStormLMConfigs)
        mock_logging_wrapper = MagicMock(spec=LoggingWrapper)
        mock_callback_handler = MagicMock()

        # Create RunnerArgument with rag_only_baseline_mode set to True
        runner_args = RunnerArgument(
            topic="Test Topic",
            rag_only_baseline_mode=True
        )

        # Initialize CoStormRunner
        runner = CoStormRunner(
            lm_config=mock_lm_config,
            runner_argument=runner_args,
            logging_wrapper=mock_logging_wrapper,
            callback_handler=mock_callback_handler
        )

        # Mock the pure_rag_agent's generate_topic_background method
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
        self.assertEqual(runner.knowledge_base.topic, "Test Topic")
        self.assertEqual(len(runner.conversation_history), 1)
        self.assertEqual(runner.conversation_history[0], mock_conv_turn)

        # Check if the knowledge base was updated correctly
        runner.knowledge_base.update_from_conv_turn.assert_called_once_with(
            conv_turn=mock_conv_turn,
            allow_create_new_node=True,
            insert_under_root=True
        )

        # Verify that the pure_rag_agent's generate_topic_background method was called
        runner.discourse_manager.pure_rag_agent.generate_topic_background.assert_called_once()

        # Verify that the logging wrapper was used
        mock_logging_wrapper.log_pipeline_stage.assert_called_once_with(pipeline_stage="warm start stage")

    @patch('knowledge_storm.storm_wiki.engine.StormOutlineGenerationModule')
    def test_run_outline_generation_module(self, mock_outline_module):
        # Set up mock objects
        mock_lm_config = MagicMock(spec=STORMWikiLMConfigs)
        mock_args = MagicMock(spec=STORMWikiRunnerArguments)
        mock_rm = MagicMock()
        mock_information_table = MagicMock(spec=StormInformationTable)
        mock_callback_handler = MagicMock(spec=BaseCallbackHandler)

        # Mock the generate_outline method
        mock_outline = MagicMock(spec=StormArticle)
        mock_draft_outline = MagicMock(spec=StormArticle)
        mock_outline_module.return_value.generate_outline.return_value = (mock_outline, mock_draft_outline)

        # Initialize STORMWikiRunner
        runner = STORMWikiRunner(args=mock_args, lm_configs=mock_lm_config, rm=mock_rm)
        runner.topic = "Test Topic"
        runner.article_output_dir = "/tmp/test_output"

        # Call run_outline_generation_module
        result = runner.run_outline_generation_module(
            information_table=mock_information_table,
            callback_handler=mock_callback_handler
        )

        # Assertions
        mock_outline_module.return_value.generate_outline.assert_called_once_with(
            topic="Test Topic",
            information_table=mock_information_table,
            return_draft_outline=True,
            callback_handler=mock_callback_handler
        )
        mock_outline.dump_outline_to_file.assert_called_once_with("/tmp/test_output/storm_gen_outline.txt")
        mock_draft_outline.dump_outline_to_file.assert_called_once_with("/tmp/test_output/direct_gen_outline.txt")
        self.assertEqual(result, mock_outline)
