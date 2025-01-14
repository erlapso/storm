import unittest

from knowledge_storm.collaborative_storm.modules.callback import LocalConsolePrintCallBackHandler
from unittest.mock import patch

class TestLocalConsolePrintCallBackHandler(unittest.TestCase):
    def setUp(self):
        self.handler = LocalConsolePrintCallBackHandler()

    @patch('builtins.print')
    def test_on_article_generation_start(self, mock_print):
        self.handler.on_article_generation_start()
        mock_print.assert_called_once_with("Start generating article.")

if __name__ == '__main__':
    unittest.main()