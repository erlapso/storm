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