import pytest

from knowledge_storm.collaborative_storm.modules.collaborative_storm_utils import extract_storm_info_snippet
from knowledge_storm.interface import Information

class TestCollaborativeStormUtils:
    def test_extract_storm_info_snippet(self):
        # Create a mock Information object
        original_info = Information(
            url="https://example.com",
            description="Test description",
            snippets=["Snippet 1", "Snippet 2", "Snippet 3"],
            title="Test Title",
            meta={"key": "value"}
        )

        # Test extracting the second snippet (index 1)
        result = extract_storm_info_snippet(original_info, 1)

        # Assert that the result is an Information object
        assert isinstance(result, Information)

        # Assert that the extracted snippet is correct
        assert result.snippets == ["Snippet 2"]

        # Assert that other attributes are preserved
        assert result.url == original_info.url
        assert result.description == original_info.description
        assert result.title == original_info.title
        assert result.meta == original_info.meta

        # Test with an out-of-range index
        with pytest.raises(ValueError):
            extract_storm_info_snippet(original_info, 3)