from scipy import sparse

def argsort(self, reverse=False):
    assert self.shape[0] == 1
    args = self.nonzero()[1]
    return sorted(args, key=lambda arg: self[0, arg], reverse=reverse)

sparse.spmatrix.argsort = argsort
