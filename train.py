import argparse
from pathlib import Path
from typing import List, Callable
import os
from enum import Enum
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from gensim.models import Word2Vec
import gensim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm


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
            with open(filepath, 'r+') as file:
                texts.append(file.read())

        return [self._preprocess(text) for text in texts]


class Pooler(Enum):
    MIN = np.min
    MEAN = np.mean
    MAX = np.max


class Word2VecVectorizer:
    def __init__(self, pooler: Callable = np.mean, load_model: bool = False):
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
        if load_model:
            self._vectorizer = gensim.models.Word2Vec.load('word2vec.model').wv
            self.vocab = self._vectorizer.key_to_index
        else:
            self._vectorizer = None
            self.vocab = None
        self._pooler = pooler

    def train(self, vocab_data, train_data, save_model: bool = True):
        self._word2vec.build_vocab(vocab_data)
        self._word2vec.train(train_data, total_examples=self._word2vec.corpus_count, epochs=300)
        if save_model:
            self._word2vec.save('word2vec.model')
            self._vectorizer = gensim.models.Word2Vec.load('word2vec.model').wv
            self.vocab = self._vectorizer.key_to_index

    def get_vector(self, word: str) -> np.array:
        return self._vectorizer.get_vector(word)

    def get_text_vector(self, text: List[str]) -> np.array:
        text_vector = []
        for word in text:
            text_vector.append(self.get_vector(word))
        text_vector = np.array(text_vector)
        return self._pooler(text_vector, axis=0)


class TripletDataset(Dataset):
    def __init__(self, paths: List[Path], data: List[List[str]], vectorizer: Callable):
        super().__init__()
        self._paths = paths
        self._vectorizer = vectorizer
        self._data = data

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx):
        anchor_path = self._paths[idx]
        anchor_folder = anchor_path.split('/')[1]
        anchor_filename = anchor_path.split('/')[-1]

        positive_path = []
        for path in self._paths:
            folder, filename = path.split('/')[1], path.split('/')[-1]
            if folder != anchor_folder and filename == anchor_filename:
                positive_path.append(path)
        positive_path = np.random.choice(positive_path)
        positive_idx = self._paths.index(positive_path)

        negative_path = np.random.choice(self._paths)
        while negative_path.split('/')[-1] == anchor_filename:
            negative_path = np.random.choice(self._paths)
        negative_idx = self._paths.index(negative_path)

        anchor = self._data[idx]
        positive = self._data[positive_idx]
        negative = self._data[negative_idx]

        anchor = self._vectorizer.get_text_vector(anchor)
        positive = self._vectorizer.get_text_vector(positive)
        negative = self._vectorizer.get_text_vector(negative)

        return anchor, positive, negative


class TripletModel(nn.Module):
    def __init__(self, in_channels: int = 300, out_channels: int = 150):
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(in_channels, in_channels - (in_channels // 6)),
            nn.Linear(in_channels - (in_channels // 6), out_channels)
        )

    def forward(self, X):
        return self._model(X)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Antiplagiat')
    parser.add_argument('files', type=str, help='original files')
    parser.add_argument('plagiat1', type=str, help='plagiat1 files')
    parser.add_argument('plagiat2', type=str, help='plagiat2 files')
    parser.add_argument('--model', type=str, help='path for saving model')
    args = parser.parse_args()

    preprocessor = Preprocessor(True)
    paths = []
    for dirs in [args.files, args.plagiat1, args.plagiat2]:
        for root, _, files in os.walk(os.path.join('data', dirs), topdown=False):
            for name in files:
                paths.append(os.path.join(root, name))
    clean_data = preprocessor.preprocess(paths)

    w2v = Word2VecVectorizer(Pooler.MEAN)
    w2v.train(clean_data, clean_data)

    dataset = TripletDataset(paths, clean_data, w2v)
    dataloader = DataLoader(dataset, batch_size=1)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y),
        margin=1.0)

    for epoch in tqdm(range(300)):
        for anchor, positive, negative in dataloader:
            optimizer.zero_grad()

            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            if anchor.shape != positive.shape or \
                anchor.shape != negative.shape or \
                    positive.shape != negative.shape:
                vector_shape = max(anchor.shape, positive.shape, negative.shape)
                if anchor.shape != vector_shape:
                    anchor = torch.zeros(vector_shape, requires_grad=True).to(DEVICE)
                if positive.shape != vector_shape:
                    positive = torch.zeros(vector_shape, requires_grad=True).to(DEVICE)
                if negative.shape != vector_shape:
                    negative = torch.zeros(vector_shape, requires_grad=True).to(DEVICE)

            anchor = model.forward(anchor)
            positive = model.forward(positive)
            negative = model.forward(negative)

            loss = criterion(anchor, positive, negative)
            loss.backward()

            optimizer.step()

            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, 'checkpoint.pkl')

    torch.save(model, args.model)
