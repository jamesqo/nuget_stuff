import logging

from scipy import sparse

from utils.logging import StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

class ChunkManager(object):
    def __init__(self, fname_fmt):
        self.fname_fmt = fname_fmt

    def save(self, chunkno, feats):
        assert sparse.isspmatrix_csr(feats)

        fname = self.fname_fmt.format(chunkno)
        LOG.debug("Saving vectors for chunk #{} to {}".format(chunkno, fname))
        sparse.save_npz(fname, feats)

    def load(self, chunkno):
        fname = self.fname_fmt.format(chunkno)
        return sparse.load_npz(fname)
