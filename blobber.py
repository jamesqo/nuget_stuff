import logging
import os

from ml import FeatureTransformer, Recommender
from scipy import sparse

from serializers import RecSerializer
from utils.logging import log_call, StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

def get_chunk(df, chunkno):
    return df[df['chunkno'] == chunkno]

def get_page(df, pageno):
    return df[df['pageno'] == pageno]

def get_chunknos(df):
    assert all(~df['chunkno'].isna())
    return sorted(set(df['chunkno']))

def get_pagenos(df):
    assert all(~df['pageno'].isna())
    return sorted(set(df['pageno']))

def gen_blob(DF, pageno, pagedf, pagefeats, blobs_root, getvecs):
    LOG.debug("Generating blobs for page #{}".format(pageno))

    M, m = DF.shape[0], pagedf.shape[0] # Good
    magic = Recommender(n_recs=5,
                        mode='chunked',
                        n_total=M,
                        n_pred=m)

    for chunkno in get_chunknos(DF):
        df_fit, feats_fit = get_chunk(DF, chunkno), getvecs(chunkno)

        magic.partial_fit(X=feats_fit,
                          df=df_fit,
                          X_pred=pagefeats,
                          df_pred=pagedf)

    recs_dict = magic.predict(pagefeats, pagedf)

    dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
    os.makedirs(dirname, exist_ok=True)

    ids = list(pagedf['id'])
    for id_ in ids:
        hexid = id_.encode('utf-8').hex()
        blob_fname = os.path.join(dirname, '{}.json'.format(hexid))

        recs = recs_dict[id_]
        writer = RecSerializer(blob_fname)
        writer.writerecs(id_, recs)

def gen_blobs(df, tagger, args, blobs_root, vectors_root):
    VEC_FMT = os.path.join(vectors_root, 'chunk{chunkno}.npz')

    # TODO: Refactor so that FeatureTransformer.fit_transform() returns 'ChunkReference' objects.
    def getvecs(chunkno):
        return sparse.load_npz(VEC_FMT.format(chunkno=chunkno))

    log_call()
    os.makedirs(blobs_root, exist_ok=True)
    os.makedirs(vectors_root, exist_ok=True)

    trans = FeatureTransformer(tags_vocab=tagger.vocab_)

    if not args.reuse_vectors:
        trans2 = FeatureTransformer(tags_vocab=tagger.vocab_,
                                    mode='chunked',
                                    output_fmt=VEC_FMT)
        fnames = trans2.fit_transform(df)
        assert len(fnames) == len(get_chunknos(df))

    pagenos = get_pagenos(df)

    for pageno in pagenos:
        pagedf = get_page(df, pageno)
        pagefeats = trans.fit_transform(pagedf)
        gen_blob(DF=df,
                 pageno=pageno,
                 pagedf=pagedf,
                 pagefeats=pagefeats,
                 blobs_root=blobs_root,
                 getvecs=getvecs)
