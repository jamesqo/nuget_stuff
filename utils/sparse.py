from scipy import sparse

def argsort(self, reverse=False):
    assert self.shape[0] == 1

    coo = self.tocoo()
    nnz, col, data = coo.getnnz(), coo.col, coo.data

    indices = sorted(range(nnz), key=data.__getitem__, reverse=reverse)
    return [col[i] for i in indices]

sparse.spmatrix.argsort = argsort
