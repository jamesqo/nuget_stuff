import logging
import os

from ml import FeatureTransformer, Recommender
from scipy import sparse

from serializers import RecSerializer

from utils.logging import log_call, StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

def gen_blobs(df, tagger, blobs_root, vectors_root):
    FNAME_FMT = os.path.join(vectors_root, 'chunk{chunkno}.npz')

    log_call()

    os.makedirs(blobs_root, exist_ok=True)
    os.makedirs(vectors_root, exist_ok=True)

    trans = FeatureTransformer(tags_vocab=tagger.vocab_,
                               mode='chunked',
                               fname_fmt=FNAME_FMT)
    fnames = trans.fit_transform(df)

    assert all(~df['chunkno'].isna())
    chunknos = sorted(set(df['chunkno']))
    assert len(chunknos) == len(fnames)

    for chunkno_pred in chunknos:
        magic = Recommender(n_recs=5)

        subdf_pred = df[df['chunkno'] == chunkno_pred]
        fname_pred = FNAME_FMT.format(chunkno=chunkno_pred)
        subfeats_pred = sparse.load_npz(fname_pred)

        for chunkno in chunknos:
            if chunkno == chunkno_pred:
                subdf, fname, subfeats = subdf_pred, fname_pred, subfeats_pred
            else:
                subdf = df[df['chunkno'] == chunkno]
                fname = FNAME_FMT.format(chunkno=chunkno)
                subfeats_pred = sparse.load_npz(fname)

            magic.partial_fit(subfeats, subdf, subfeats_pred, subdf_pred)

        recs_dict = magic.predict(subfeats_pred, subdf_pred)

        pagenos = sorted(set(subdf_pred['pageno']))
        for pageno in pagenos:
            subsubdf = subdf_pred[subdf_pred['pageno'] == pageno]
            dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
            os.makedirs(dirname, exist_ok=True)

            ids = list(subsubdf['id'])
            for id_ in ids:
                hexid = id_.encode('utf-8').hex()
                blob_fname = os.path.join(dirname, '{}.json'.format(hexid))

                recs = recs_dict[id_]
                writer = RecSerializer(blob_fname)
                writer.writerecs(id_, recs)
