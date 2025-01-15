import unittest

from knowledge_storm.utils import ArticleTextProcessing

class TestArticleTextProcessing(unittest.TestCase):
    def test_clean_up_outline(self):
        sample_outline = """
# Main Topic

## Introduction

## Key Points

## See also
- Related topic 1
- Related topic 2

## References
1. Reference 1
2. Reference 2

## External links
- Link 1
- Link 2

## Conclusion
"""
        expected_cleaned_outline = """
# Main Topic

## Introduction

## Key Points

## Conclusion
"""
        cleaned_outline = ArticleTextProcessing.clean_up_outline(sample_outline)
        self.assertEqual(cleaned_outline.strip(), expected_cleaned_outline.strip())