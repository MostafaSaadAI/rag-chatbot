import faiss
import numpy as np

class FAISSStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)

    def add_vectors(self, vectors):
        self.index.add(vectors)

    def search(self, query_vector, top_k=3):
        query_vector = np.array([query_vector])
        distances, indices = self.index.search(query_vector, top_k)
        return distances, indices
