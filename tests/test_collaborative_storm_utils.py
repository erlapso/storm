import pytest

from knowledge_storm.collaborative_storm.modules.collaborative_storm_utils import format_search_results
from knowledge_storm.interface import Information
from typing import List

class TestCollaborativeStormUtils:
    def test_format_search_results_brief_mode(self):
        # Create mock Information objects
        info1 = Information(
            url="http://example1.com",
            description="Example 1",
            snippets=["Snippet 1 for example 1", "Snippet 2 for example 1"],
            title="Example 1 Title",
            meta={}
        )
        info2 = Information(
            url="http://example2.com",
            description="Example 2",
            snippets=["Snippet 1 for example 2", "Snippet 2 for example 2"],
            title="Example 2 Title",
            meta={}
        )

        searched_results = [info1, info2]

        # Call the function with brief mode
        formatted_results, index_mapping = format_search_results(
            searched_results,
            info_max_num_words=100,
            mode="brief"
        )

        # Assert the correct formatting
        expected_output = "[1]: Snippet 1 for example 1\n[2]: Snippet 1 for example 2"
        assert formatted_results == expected_output

        # Assert the correct index mapping
        assert len(index_mapping) == 2
        assert index_mapping[1].url == "http://example1.com"
        assert index_mapping[2].url == "http://example2.com"

        # Assert that only the first snippet is included for each Information object
        assert len(index_mapping[1].snippets) == 1
        assert len(index_mapping[2].snippets) == 1
        assert index_mapping[1].snippets[0] == "Snippet 1 for example 1"
        assert index_mapping[2].snippets[0] == "Snippet 1 for example 2"