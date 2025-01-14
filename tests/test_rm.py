import os
import unittest

from knowledge_storm.rm import YouRM
from unittest.mock import patch

class TestYouRM(unittest.TestCase):
    def test_initialization(self):
        # Test initialization with API key provided
        rm = YouRM(ydc_api_key="test_key")
        self.assertEqual(rm.ydc_api_key, "test_key")

        # Test initialization with environment variable
        with patch.dict(os.environ, {"YDC_API_KEY": "env_key"}):
            rm = YouRM()
            self.assertEqual(rm.ydc_api_key, "env_key")

        # Test initialization without API key (should raise an exception)
        with self.assertRaises(RuntimeError):
            YouRM()

    def test_get_usage_and_reset(self):
        rm = YouRM(ydc_api_key="test_key")

        # Simulate some usage
        rm.usage = 5

        # Check usage and reset
        usage = rm.get_usage_and_reset()
        self.assertEqual(usage, {"YouRM": 5})

        # Check if usage was reset
        self.assertEqual(rm.usage, 0)

        # Check usage again (should be 0)
        usage = rm.get_usage_and_reset()
        self.assertEqual(usage, {"YouRM": 0})