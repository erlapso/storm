import unittest

from knowledge_storm.utils import ArticleTextProcessing, truncate_filename

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

class TestArticleTextProcessing(unittest.TestCase):
    def test_remove_citations(self):
        # Test with single citation
        text_with_single_citation = "This is a sentence with a citation[1]."
        self.assertEqual(ArticleTextProcessing.remove_citations(text_with_single_citation), 
                         "This is a sentence with a citation.")

        # Test with multiple citations
        text_with_multiple_citations = "This sentence has two citations[1][2]."
        self.assertEqual(ArticleTextProcessing.remove_citations(text_with_multiple_citations), 
                         "This sentence has two citations.")

        # Test with citations containing multiple numbers
        text_with_complex_citations = "This is a complex citation[1,2,3]."
        self.assertEqual(ArticleTextProcessing.remove_citations(text_with_complex_citations), 
                         "This is a complex citation.")

        # Test with no citations
        text_without_citations = "This sentence has no citations."
        self.assertEqual(ArticleTextProcessing.remove_citations(text_without_citations), 
                         text_without_citations)

        # Test with mixed content
        mixed_text = "Citation[1] in the middle and [2,3] at the end."
        self.assertEqual(ArticleTextProcessing.remove_citations(mixed_text), 
                         "Citation in the middle and  at the end.")

    def test_parse_citation_indices(self):
        # Test with a single citation
        single_citation = "This is a sentence with a single citation [1]."
        self.assertEqual(ArticleTextProcessing.parse_citation_indices(single_citation), [1])

        # Test with multiple citations
        multiple_citations = "This sentence has multiple citations [1] [2] [3]."
        self.assertEqual(ArticleTextProcessing.parse_citation_indices(multiple_citations), [1, 2, 3])

        # Test with out-of-order citations
        out_of_order = "Citations can be [2] out of [1] order [3]."
        self.assertEqual(ArticleTextProcessing.parse_citation_indices(out_of_order), [2, 1, 3])

        # Test with repeated citations
        repeated_citations = "This [1] citation [2] is repeated [1] [2]."
        self.assertEqual(ArticleTextProcessing.parse_citation_indices(repeated_citations), [1, 2])

        # Test with no citations
        no_citations = "This sentence has no citations."
        self.assertEqual(ArticleTextProcessing.parse_citation_indices(no_citations), [])

        # Test with non-numeric content in brackets
        non_numeric = "This [one] is not a [valid] citation, but [1] is."
        self.assertEqual(ArticleTextProcessing.parse_citation_indices(non_numeric), [1])

    def test_remove_uncompleted_sentences_with_citations(self):
        # Test with complete sentences and citations
        text = "This is a complete sentence [1]. This is another one [2]."
        self.assertEqual(ArticleTextProcessing.remove_uncompleted_sentences_with_citations(text), text)

        # Test with an incomplete sentence at the end
        text = "This is a complete sentence [1]. This is incomplete"
        expected = "This is a complete sentence [1]."
        self.assertEqual(ArticleTextProcessing.remove_uncompleted_sentences_with_citations(text), expected)

        # Test with multiple citations
        text = "This is a sentence [1][2]. This is incomplete [3]"
        expected = "This is a sentence [1][2]."
        self.assertEqual(ArticleTextProcessing.remove_uncompleted_sentences_with_citations(text), expected)

        # Test with grouped citations
        text = "This is a sentence [1, 2, 3]. This is another one [4, 5]."
        expected = "This is a sentence [1][2][3]. This is another one [4][5]."
        self.assertEqual(ArticleTextProcessing.remove_uncompleted_sentences_with_citations(text), expected)

        # Test with an empty string
        self.assertEqual(ArticleTextProcessing.remove_uncompleted_sentences_with_citations(""), "")

    def test_clean_up_outline(self):
        # Sample outline with sections to be removed and citations
        outline = """
# Introduction

# Main Content
## Section 1
- Point 1 [1]
- Point 2 [2]

## Section 2
- Another point [3]

# See also
Some related topics

# References
1. Reference 1
2. Reference 2
3. Reference 3

# External links
Some external links
"""
        expected_outline = """
# Introduction

# Main Content
## Section 1
- Point 1
- Point 2

## Section 2
- Another point
"""
        cleaned_outline = ArticleTextProcessing.clean_up_outline(outline)

        # Remove leading/trailing whitespace and compare
        self.assertEqual(cleaned_outline.strip(), expected_outline.strip())

        # Test with a specific topic
        topic = "Main Content"
        cleaned_outline_with_topic = ArticleTextProcessing.clean_up_outline(outline, topic)
        expected_outline_with_topic = """
# Main Content
## Section 1
- Point 1
- Point 2

## Section 2
- Another point
"""
        self.assertEqual(cleaned_outline_with_topic.strip(), expected_outline_with_topic.strip())

    def test_clean_up_section(self):
        # Test input with uncompleted sentences, duplicate citations, and unnecessary summary
        input_text = """
This is a complete sentence. [1] This is another complete sentence. [1][2]
This is an incomplete sentence
# Summary
This summary should be removed.
Overall, this is a conclusion.
# New Section
This is a new section. [3]
"""
        expected_output = """
This is a complete sentence. [1] This is another complete sentence. [1][2]

# New Section
This is a new section. [3]"""

        cleaned_text = ArticleTextProcessing.clean_up_section(input_text)
        self.assertEqual(cleaned_text.strip(), expected_output.strip())

        # Test with no changes needed
        no_change_text = "This is a complete sentence. [1]"
        self.assertEqual(ArticleTextProcessing.clean_up_section(no_change_text).strip(), no_change_text.strip())

        # Test with only summary removal
        summary_text = "This is a sentence. [1]\n# Summary\nThis should be removed."
        expected_summary_output = "This is a sentence. [1]"
        self.assertEqual(ArticleTextProcessing.clean_up_section(summary_text).strip(), expected_summary_output.strip())
