from pathlib import Path
from typing import List
from nltk.tokenize import WordPunctTokenizer


class Preprocessor:

    def __init__(self, do_tokenize: bool = True):
        self._do_tokenize = do_tokenize
        self._tokenizer = WordPunctTokenizer()

    def _preprocess(self, text: List[str]) -> List[str]:
        text = text.lower()
        tokens = text.split(' ')
        if self._do_tokenize:
            tokens = self._tokenizer.tokenize(text)
        return tokens

    def preprocess(self, filepaths: List[Path]) -> List[str]:
        texts = []
        for filepath in filepaths:
            with open(filepath, 'r') as file:
                texts.append(file.read())

        return [self._preprocess(text) for text in texts]
