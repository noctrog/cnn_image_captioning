import torch

class MiniGlove():
    def __init__(self, vectors=None, stoi=None, itos=None):
        self.vectors = vectors
        self.stoi = stoi
        self.itos = itos

        if self.vectors is not None:
            self.dim = self.vectors.shape[1]

    def load(self):
        state = torch.load('./weights/custom_embedding.dat')
        self.vectors = state['vectors']
        self.dim = state['dim']
        self.stoi = state['stoi']
        self.itos = state['itos']
        pass

    def save(self):
        state = {'vectors': self.vectors,
                 'dim': self.dim,
                 'stoi': self.stoi,
                 'itos': self.itos}
        torch.save(state, './weights/custom_embedding.dat')

    def _set_vectors(self, new_vectors):
        self.vectors = new_vectors

    def _set_stoi(self, new_stoi):
        self.stoi = new_stoi

    def _set_itos(self, new_itos):
        self.itos = new_itos
