import numpy as np


class GloVe():
    def __init__(self, vectors_file):
        with open(vectors_file, "r") as f:
            vectors = {}
            for line in f:
                vec = line.rstrip().split(" ")
                vectors[vec[0]] = np.array([float(x) for x in vec[1:]],
                                           dtype=np.float64)

        self.vocab_size = len(vectors.items())
        self.vocabi = {w: idx for idx, w in enumerate(vectors.keys())}
        self.ivocab = {idx: w for idx, w in enumerate(vectors.keys())}
        self.hidden_dim = vectors[self.ivocab[0]].shape[0]
        self.W = np.zeros((self.vocab_size, self.hidden_dim))
        for word, v in vectors.items():
            if word == '<unk>':
                continue
            self.W[self.vocabi[word], :] = v

    def get_vector_seq(self, text):
        vectors = []
        for word in text.split(" "):
            if word in self.vocabi.keys():
                vectors.append(self.W[self.vocabi[word]])
        return np.array(vectors, dtype=np.float64)


if __name__ == "__main__":
    app = GloVe(
        vectors_file="/GloVe/pretrained/glove.6B.50d.txt"
    )
