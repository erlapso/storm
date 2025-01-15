import pytz
import unittest

from contextlib import contextmanager
from datetime import datetime, timedelta
from knowledge_storm.logging_wrapper import CALIFORNIA_TZ, LoggingWrapper
from unittest.mock import Mock, patch

class TestLoggingWrapper(unittest.TestCase):
    def setUp(self):
        self.mock_lm_config = Mock()
        self.mock_lm_config.collect_and_reset_lm_usage.return_value = {}
        self.mock_lm_config.collect_and_reset_lm_history.return_value = []
        self.logging_wrapper = LoggingWrapper(self.mock_lm_config)

    @patch('knowledge_storm.logging_wrapper.datetime')
    def test_nested_event_logging(self, mock_datetime):
        # Set up mock datetime to return predictable values
        mock_now = datetime(2023, 1, 1, 12, 0, 0, tzinfo=pytz.utc)
        mock_datetime.now.side_effect = [
            mock_now,
            mock_now + timedelta(seconds=1),
            mock_now + timedelta(seconds=2),
            mock_now + timedelta(seconds=3),
            mock_now + timedelta(seconds=4),
            mock_now + timedelta(seconds=5),
        ]

        with self.logging_wrapper.log_pipeline_stage("test_stage"):
            with self.logging_wrapper.log_event("parent_event"):
                with self.logging_wrapper.log_event("child_event"):
                    pass

        log_dump = self.logging_wrapper.dump_logging_and_reset()

        # Check if the stage and events are recorded
        self.assertIn("test_stage", log_dump)
        self.assertIn("parent_event", log_dump["test_stage"]["time_usage"])

        parent_event = log_dump["test_stage"]["time_usage"]["parent_event"]
        self.assertIn("child_event", parent_event.get_child_events())

        # Check timings
        self.assertEqual(parent_event.get_total_time(), 4)
        self.assertEqual(parent_event.get_child_events()["child_event"].get_total_time(), 2)

        # Check formatted times
        self.assertEqual(parent_event.get_start_time(), "2023-01-01 04:00:00.000")
        self.assertEqual(parent_event.get_end_time(), "2023-01-01 04:00:04.000")
        self.assertEqual(parent_event.get_child_events()["child_event"].get_start_time(), "2023-01-01 04:00:01.000")
        self.assertEqual(parent_event.get_child_events()["child_event"].get_end_time(), "2023-01-01 04:00:03.000")