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

def gen_blob(DF, chunkno, df, feats, blobs_root, getvecs):
    M, m = DF.shape[0], df.shape[0] # Good
    magic = Recommender(n_recs=5,
                        mode='chunked',
                        n_total=M,
                        n_pred=m)

    for chunkno2 in get_chunknos(DF):
        if chunkno2 == chunkno:
            df_fit, feats_fit = df, feats
        else:
            df_fit, feats_fit = get_chunk(DF, chunkno2), getvecs(chunkno2)

        magic.partial_fit(X=feats_fit,
                          df=df_fit,
                          X_pred=feats,
                          df_pred=df)

    recs_dict = magic.predict(feats, df)

    pagenos = sorted(set(df['pageno']))
    for pageno in pagenos:
        pagedf = get_page(df, pageno)
        dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
        os.makedirs(dirname, exist_ok=True)

        ids = list(pagedf['id'])
        for id_ in ids:
            hexid = id_.encode('utf-8').hex()
            blob_fname = os.path.join(dirname, '{}.json'.format(hexid))

            recs = recs_dict[id_]
            writer = RecSerializer(blob_fname)
            writer.writerecs(id_, recs)

def gen_blobs(df, tagger, blobs_root, vectors_root):
    VEC_FMT = os.path.join(vectors_root, 'chunk{chunkno}.npz')

    def getvecs(chunkno):
        return sparse.load_npz(VEC_FMT.format(chunkno=chunkno))

    log_call()
    os.makedirs(blobs_root, exist_ok=True)
    os.makedirs(vectors_root, exist_ok=True)

    trans = FeatureTransformer(tags_vocab=tagger.vocab_,
                               mode='chunked',
                               output_fmt=VEC_FMT)
    fnames = trans.fit_transform(df)

    chunknos = get_chunknos(df)
    assert len(chunknos) == len(fnames)

    for chunkno in chunknos:
        chunkdf, chunkfeats = get_chunk(df, chunkno), getvecs(chunkno)
        gen_blob(DF=df,
                 chunkno=chunkno,
                 df=chunkdf,
                 feats=chunkfeats,
                 blobs_root=blobs_root,
                 getvecs=getvecs)
