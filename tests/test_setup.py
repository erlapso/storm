import pytest
import re

from setuptools import setup
from unittest.mock import mock_open, patch

class TestSetup:
    @patch('builtins.open', new_callable=mock_open, read_data='<p>Test content</p>\nMore content')
    def test_long_description_processing(self, mock_file):
        # This test checks if the long_description is correctly processed
        # by removing <p> tags using regex

        # Mock the requirements.txt content
        mock_file.side_effect = [
            mock_open(read_data='<p>Test content</p>\nMore content').return_value,
            mock_open(read_data='requirement1\nrequirement2').return_value
        ]

        # Call the setup function (this will use our mocked open function)
        with patch('setuptools.setup') as mock_setup:
            import setup

        # Check if setup was called
        mock_setup.assert_called_once()

        # Get the arguments passed to setup
        setup_args = mock_setup.call_args[1]

        # Check if <p> tags were removed from long_description
        assert '<p>' not in setup_args['long_description']
        assert '</p>' not in setup_args['long_description']
        assert 'Test content' not in setup_args['long_description']
        assert 'More content' in setup_args['long_description']

        # Check if requirements were correctly read
        assert setup_args['install_requires'] == ['requirement1', 'requirement2']