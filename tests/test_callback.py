import unittest

from knowledge_storm.collaborative_storm.modules.callback import LocalConsolePrintCallBackHandler
from knowledge_storm.storm_wiki.modules.callback import BaseCallbackHandler
from unittest.mock import Mock, patch

class TestLocalConsolePrintCallBackHandler(unittest.TestCase):
    def setUp(self):
        self.handler = LocalConsolePrintCallBackHandler()

    @patch('builtins.print')
    def test_on_article_generation_start(self, mock_print):
        self.handler.on_article_generation_start()
        mock_print.assert_called_once_with("Start generating article.")

if __name__ == '__main__':
    unittest.main()

    def test_on_identify_perspective_end(self):
        handler = BaseCallbackHandler()
        mock_perspectives = Mock()
        handler.on_identify_perspective_end(perspectives=mock_perspectives)
        # Since the method is empty in the base class, we just verify it can be called without errors
        self.assertTrue(True)