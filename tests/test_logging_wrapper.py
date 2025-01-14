import time
import unittest

from knowledge_storm.logging_wrapper import EventLog, LoggingWrapper
from unittest.mock import MagicMock

class TestLoggingWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_lm_config = MagicMock()
        self.logging_wrapper = LoggingWrapper(self.mock_lm_config)

    def test_nested_event_logging(self):
        with self.logging_wrapper.log_pipeline_stage("test_stage"):
            with self.logging_wrapper.log_event("parent_event"):
                time.sleep(0.1)  # Simulate some work
                with self.logging_wrapper.log_event("child_event"):
                    time.sleep(0.05)  # Simulate nested work

        log_dump = self.logging_wrapper.dump_logging_and_reset()

        # Check if the pipeline stage exists
        self.assertIn("test_stage", log_dump)

        # Check if both events are recorded
        self.assertIn("parent_event", log_dump["test_stage"]["time_usage"])
        self.assertIn("child_event", log_dump["test_stage"]["time_usage"])

        # Check if timings are reasonable
        parent_time = log_dump["test_stage"]["time_usage"]["parent_event"]["total_time_seconds"]
        child_time = log_dump["test_stage"]["time_usage"]["child_event"]["total_time_seconds"]

        self.assertGreater(parent_time, child_time)
        self.assertGreaterEqual(parent_time, 0.15)  # At least 0.15 seconds (0.1 + 0.05)
        self.assertGreaterEqual(child_time, 0.05)  # At least 0.05 seconds

        # Check if child event is nested under parent event
        parent_event = self.logging_wrapper.logging_dict["test_stage"]["time_usage"]["parent_event"]
        self.assertIn("child_event", parent_event.get_child_events())

    def test_add_query_count(self):
        # Test adding query count within a pipeline stage
        with self.logging_wrapper.log_pipeline_stage("query_stage"):
            self.logging_wrapper.add_query_count(3)
            self.logging_wrapper.add_query_count(2)
            self.logging_wrapper.add_query_count(1)

        log_dump = self.logging_wrapper.dump_logging_and_reset()

        # Check if the query count is correctly recorded
        self.assertIn("query_stage", log_dump)
        self.assertEqual(log_dump["query_stage"]["query_count"], 6)

        # Test adding query count outside of a pipeline stage
        with self.assertRaises(RuntimeError):
            self.logging_wrapper.add_query_count(1)

    def test_start_new_pipeline_stage_while_active(self):
        # Create a mock LM config
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Start the first pipeline stage
        with logging_wrapper.log_pipeline_stage("stage1"):
            # Attempt to start a new pipeline stage while the first one is still active
            with logging_wrapper.log_pipeline_stage("stage2"):
                pass

        # Check if both stages are recorded in the logging dictionary
        log_dump = logging_wrapper.dump_logging_and_reset()
        self.assertIn("stage1", log_dump)
        self.assertIn("stage2", log_dump)

        # Check if stage1 has a total_wall_time (indicating it was properly ended)
        self.assertIn("total_wall_time", log_dump["stage1"])

        # Check if stage2 also has a total_wall_time
        self.assertIn("total_wall_time", log_dump["stage2"])

    def test_end_nonexistent_event(self):
        # Create a LoggingWrapper instance
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Start a pipeline stage
        with logging_wrapper.log_pipeline_stage("test_stage"):
            # Attempt to end an event that wasn't started
            with self.assertRaises(RuntimeError):
                logging_wrapper._event_end("nonexistent_event")

    def test_start_event_without_active_pipeline_stage(self):
        # Create a LoggingWrapper instance
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Attempt to start an event without an active pipeline stage
        with self.assertRaises(RuntimeError) as context:
            logging_wrapper._event_start("test_event")

        # Check if the error message is correct
        self.assertEqual(
            str(context.exception),
            "No pipeline stage is currently active."
        )

    def test_end_nonexistent_pipeline_stage(self):
        # Create a LoggingWrapper instance
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Attempt to end a pipeline stage that wasn't started
        with self.assertRaises(RuntimeError) as context:
            logging_wrapper._pipeline_stage_end()

        # Check if the error message is correct
        self.assertEqual(
            str(context.exception),
            "No pipeline stage is currently active to end."
        )

    def test_start_nested_event_without_parent(self):
        # Create a LoggingWrapper instance
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Start a pipeline stage
        with logging_wrapper.log_pipeline_stage("test_stage"):
            # Clear the event stack
            logging_wrapper.event_stack.clear()

            # Attempt to start a nested event without an active parent event
            with self.assertRaises(RuntimeError) as context:
                logging_wrapper._event_start("nested_event")

            # Check if the error message is correct
            self.assertEqual(
                str(context.exception),
                "Cannot start an event without an active pipeline stage or parent event."
            )

    def test_end_mismatched_event(self):
        # Create a LoggingWrapper instance
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Start a pipeline stage
        with logging_wrapper.log_pipeline_stage("test_stage"):
            # Start two nested events
            with logging_wrapper.log_event("parent_event"):
                with logging_wrapper.log_event("child_event"):
                    # Attempt to end the parent event before the child event
                    with self.assertRaises(RuntimeError) as context:
                        logging_wrapper._event_end("parent_event")

                # Check if the error message is correct
                self.assertEqual(
                    str(context.exception),
                    "Cannot end an event without an active parent event."
                )

        # Verify that both events were properly ended
        log_dump = logging_wrapper.dump_logging_and_reset()
        self.assertIn("parent_event", log_dump["test_stage"]["time_usage"])
        self.assertIn("child_event", log_dump["test_stage"]["time_usage"])
        self.assertIsNotNone(log_dump["test_stage"]["time_usage"]["parent_event"]["end_time"])
        self.assertIsNotNone(log_dump["test_stage"]["time_usage"]["child_event"]["end_time"])

    def test_add_child_to_nonexistent_parent(self):
        # Create a LoggingWrapper instance
        mock_lm_config = MagicMock()
        logging_wrapper = LoggingWrapper(mock_lm_config)

        # Start a pipeline stage
        with logging_wrapper.log_pipeline_stage("test_stage"):
            # Start a parent event
            with logging_wrapper.log_event("parent_event"):
                pass

            # Clear the event stack to simulate a non-existent parent
            logging_wrapper.event_stack.clear()

            # Attempt to start a child event without an active parent
            with self.assertRaises(RuntimeError) as context:
                logging_wrapper._event_start("child_event")

            # Check if the error message is correct
            self.assertEqual(
                str(context.exception),
                "Cannot start an event without an active pipeline stage or parent event."
            )

        # Verify that only the parent event was recorded
        log_dump = logging_wrapper.dump_logging_and_reset()
        self.assertIn("parent_event", log_dump["test_stage"]["time_usage"])
        self.assertNotIn("child_event", log_dump["test_stage"]["time_usage"])

def test_add_query_count_without_active_pipeline_stage(self):
    # Create a new LoggingWrapper instance
    mock_lm_config = MagicMock()
    logging_wrapper = LoggingWrapper(mock_lm_config)

    # Attempt to add a query count without an active pipeline stage
    with self.assertRaises(RuntimeError) as context:
        logging_wrapper.add_query_count(1)

    # Check if the error message is correct
    self.assertEqual(
        str(context.exception),
        "No pipeline stage is currently active to add query count."
    )