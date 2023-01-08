import argparse
import os
from pathlib import Path
from train import Preprocessor, Word2VecVectorizer, TripletModel
import torch
import numpy as np


def load_model(path: Path = 'model.pkl') -> torch.nn.Module:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TripletModel()
    model = torch.load(path, map_location=torch.device(device=DEVICE))
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Antiplagiat inference')
    parser.add_argument('input', type=str, help='input files')
    parser.add_argument('output', type=str, help='output files')
    parser.add_argument('--model', type=str, help='path for loading model')
    args = parser.parse_args()

    preprocessor = Preprocessor(True)
    w2v = Word2VecVectorizer(load_model=True)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(args.model).to(DEVICE)
    model.eval()
    with open(args.input, 'r+') as file:
        paths = file.readlines()
    for idx in range(len(paths)):
        files = paths[idx].split(' ')[:-1]
        for filename in files:
            files[files.index(filename)] = os.path.join('data', filename)
        paths[idx] = files
    outputs = []
    for files in paths:
        clean_data = preprocessor.preprocess(files)
        for text in clean_data:
            clean_text = []
            for word in text:
                if word in w2v.vocab:
                    clean_text.append(word)
            clean_data[clean_data.index(text)] = clean_text

        vec1 = torch.tensor(w2v.get_text_vector(clean_data[0])).to(DEVICE)
        vec2 = torch.tensor(w2v.get_text_vector(clean_data[1])).to(DEVICE)

        vec1 = model.forward(vec1)
        vec2 = model.forward(vec2)

        dist = torch.nn.CosineSimilarity(dim=-1)
        outputs.append(dist(vec1, vec2).item())

    np.savetxt(args.output, outputs)




