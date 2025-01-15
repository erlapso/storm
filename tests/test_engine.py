import os
import unittest

from knowledge_storm.collaborative_storm.engine import CoStormRunner, CollaborativeStormLMConfigs, RunnerArgument
from knowledge_storm.logging_wrapper import LoggingWrapper
from knowledge_storm.storm_wiki.engine import STORMWikiLMConfigs, STORMWikiRunner, STORMWikiRunnerArguments
from knowledge_storm.storm_wiki.modules.storm_dataclass import StormInformationTable
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

class TestSTORMWikiRunner(unittest.TestCase):
    @patch('knowledge_storm.storm_wiki.engine.StormKnowledgeCurationModule')
    @patch('knowledge_storm.storm_wiki.engine.FileIOHelper')
    def test_run_knowledge_curation_module(self, mock_file_io_helper, mock_knowledge_curation_module):
        # Mock the dependencies
        mock_args = MagicMock(spec=STORMWikiRunnerArguments)
        mock_lm_configs = MagicMock(spec=STORMWikiLMConfigs)
        mock_rm = MagicMock()

        # Create the STORMWikiRunner instance
        runner = STORMWikiRunner(args=mock_args, lm_configs=mock_lm_configs, rm=mock_rm)
        runner.topic = "Test Topic"
        runner.article_output_dir = "/tmp/test_output"

        # Mock the research method of StormKnowledgeCurationModule
        mock_information_table = MagicMock(spec=StormInformationTable)
        mock_conversation_log = {"log": "test"}
        mock_knowledge_curation_module.return_value.research.return_value = (mock_information_table, mock_conversation_log)

        # Call the run_knowledge_curation_module method
        result = runner.run_knowledge_curation_module(ground_truth_url="http://example.com")

        # Assertions
        self.assertEqual(result, mock_information_table)
        mock_knowledge_curation_module.return_value.research.assert_called_once_with(
            topic="Test Topic",
            ground_truth_url="http://example.com",
            callback_handler=None,
            max_perspective=mock_args.max_perspective,
            disable_perspective=False,
            return_conversation_log=True
        )
        mock_file_io_helper.dump_json.assert_called_with(
            mock_conversation_log,
            os.path.join("/tmp/test_output", "conversation_log.json")
        )
        mock_information_table.dump_url_to_info.assert_called_with(
            os.path.join("/tmp/test_output", "raw_search_results.json")
        )
