import tempfile
import unittest
from pathlib import Path
from app import DocumentIndex, index


class SearchTests(unittest.TestCase):
    def test_cited_answer(self):
        result = index.answer('What is data leakage?')
        self.assertTrue(result['sources'])
        self.assertIn('Split data before', result['answer'])
        self.assertEqual(result['sources'][0]['source'], 'ml-guide.md')

    def test_unknown_question_abstains(self):
        result = index.answer('astronauts moon cheese')
        self.assertEqual(result['sources'], [])
        self.assertIn('No supported answer', result['answer'])

    def test_invalid_request(self):
        with self.assertRaises(ValueError):
            index.answer('   ')
        with self.assertRaises(ValueError):
            index.answer('data', 0)

    def test_empty_corpus(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                DocumentIndex(Path(folder))


if __name__ == '__main__':
    unittest.main()
