from pathlib import Path
from typing import List, Callable
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
import gensim


class Preprocessor:

    def __init__(self, do_tokenize: bool = True):
        """

        Args:
            do_tokenize:
        """
        self._do_tokenize = do_tokenize
        self._tokenizer = WordPunctTokenizer()

    def _preprocess(self, text: List[str]) -> List[str]:
        """

        Args:
            text:

        Returns:

        """
        text = text.lower()
        tokens = text.split(' ')
        if self._do_tokenize:
            tokens = self._tokenizer.tokenize(text)
        return tokens

    def preprocess(self, filepaths: List[Path]) -> List[str]:
        """

        Args:
            filepaths:

        Returns:

        """
        texts = []
        for filepath in filepaths:
            with open(filepath, 'r+') as file:
                texts.append(file.read())

        return [self._preprocess(text) for text in texts]


class Word2VecVectorizer:
    def __init__(self, pooler: Callable = np.mean):
        """

        Args:
            pooler:
        """
        self._word2vec = Word2Vec(
            min_count=1,
            window=5,
            vector_size=300,
            negative=10,
            alpha=0.03,
            min_alpha=0.0007,
            sample=6e-5,
            sg=1,
            workers=4)
        self._vectorizer = None
        self._pooler = pooler

    def train(self, vocab_data, train_data, save_model: bool = True):
        self._word2vec.build_vocab(vocab_data)
        self._word2vec.train(train_data, total_examples=self._word2vec.corpus_count, epochs=1, report_delay=1)
        if save_model:
            self._word2vec.save("word2vec.model")
            self._vectorizer = gensim.models.Word2Vec.load("word2vec.model").wv

    def get_vector(self, word: str) -> np.array:
        return self._vectorizer.get_vector(word)

    def get_text_vector(self, text: List[str]) -> np.array:
        text_vector = []
        for word in text:
            text_vector.append(self.get_vector(word))
        text_vector = np.array(text_vector)
        return self._pooler(text_vector, axis = 0)