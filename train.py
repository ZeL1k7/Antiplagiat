from pathlib import Path
from typing import List, Callable
import pandas as pd
import numpy as np
from nltk.tokenize import WordPunctTokenizer


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


class FeatureCreator:
    def __init__(self, distances: Callable = None, embeddings: List[int, Callable] = None):
        """
        Creates features from texts
        Args:
            distances: measuring the distance between texts
            embeddings: shape of embedding and way to generate it
        """
        self._distances = distances
        self._embeddings = embeddings

    def _make_features(self, texts: List[List[str]]) -> pd.DataFrame:
        _features = np.array([])
        """
        Args:
            texts:

        Returns:
            DataFrame with features
        """
        if self._distances:
            features = np.empty([len(texts), len(texts)])
            for text_1 in texts:
                for text_2 in texts:
                    features[texts.index(text_1), texts.index(text_2)] = self._distances(text_1, text_2)
            _features = np.concatenate([_features, features])

        if self._embeddings:
            features = np.empty([len(texts)])
            _features = np.concatenate([_features, features])
            pass

