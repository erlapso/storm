import unittest

from knowledge_storm.utils import truncate_filename

class TestFileIOHelper(unittest.TestCase):
    def test_truncate_filename(self):
        # Test with a filename shorter than the max length
        short_filename = "short.txt"
        self.assertEqual(truncate_filename(short_filename), short_filename)

        # Test with a filename exactly at the max length
        max_length_filename = "a" * 125 + ".txt"
        self.assertEqual(truncate_filename(max_length_filename), max_length_filename)

        # Test with a filename longer than the max length
        long_filename = "a" * 130 + ".txt"
        truncated = truncate_filename(long_filename)
        self.assertEqual(len(truncated), 125)
        self.assertTrue(truncated.startswith("a" * 121))  # 125 - len(".txt")

        # Test with a custom max_length
        custom_max_length = 50
        long_filename = "a" * 60 + ".txt"
        truncated = truncate_filename(long_filename, custom_max_length)
        self.assertEqual(len(truncated), custom_max_length)
        self.assertTrue(truncated.startswith("a" * 46))  # 50 - len(".txt")