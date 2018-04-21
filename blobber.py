import logging
import os
import shutil

from chunkmgr import ChunkManager
from ml import FeatureTransformer, Recommender
from serializers import RecSerializer
from utils.path import extended_path
from utils.platform import is_windows
from utils.logging import log_call, StyleAdapter

LOG = StyleAdapter(logging.getLogger(__name__))

def get_chunk(df, chunkno):
    return df[df['chunkno'] == chunkno]

def get_page(df, pageno):
    return df[df['pageno'] == pageno]

def chunknos(df):
    assert all(~df['chunkno'].isna())
    return sorted(set(df['chunkno']))

def pagenos(df):
    assert all(~df['pageno'].isna())
    return sorted(set(df['pageno']))

def gen_blobs_for_page(pageno, df, feats, parentdf, blobs_root, chunkmgr):
    LOG.debug("Generating blobs for page #{}", pageno)

    M, m = parentdf.shape[0], df.shape[0] # Good
    magic = Recommender(n_recs=5, # TODO: This should be a command-line option
                        mode='chunked',
                        n_total=M,
                        n_pred=m)

    for chunkno in chunknos(parentdf):
        df_fit, feats_fit = get_chunk(parentdf, chunkno), chunkmgr.load(chunkno)

        magic.partial_fit(X=feats_fit,
                          df=df_fit,
                          X_pred=feats,
                          df_pred=df)

    recs_dict = magic.predict(feats, df)

    dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
    os.makedirs(dirname, exist_ok=True)

    ids = list(df['id'])
    for id_ in ids:
        hexid = id_.encode('utf-8').hex()
        blob_fname = os.path.join(dirname, '{}.json'.format(hexid))
        if is_windows:
            blob_fname = extended_path(blob_fname)

        recs = recs_dict[id_]
        writer = RecSerializer(blob_fname)
        writer.writerecs(id_, recs)

def gen_blobs(df, tagger, args, blobs_root, vectors_root):
    log_call()

    chunk_fmt = os.path.join(vectors_root, 'chunk{}.npz')
    chunkmgr = ChunkManager(chunk_fmt)

    if not args.force_refresh_vectors and os.path.isdir(vectors_root):
        LOG.debug("Using existing vectors from {}", vectors_root)
        trans = FeatureTransformer(tags_vocab=tagger.vocab_)
        trans.fit(df)
    else:
        shutil.rmtree(vectors_root, ignore_errors=True)
        os.makedirs(vectors_root, exist_ok=True)
        trans = FeatureTransformer(tags_vocab=tagger.vocab_,
                                   mode='chunked',
                                   chunkmgr=chunkmgr)
        trans.fit_transform(df)
        trans.mode = 'onego'

    shutil.rmtree(blobs_root, ignore_errors=True)
    os.makedirs(blobs_root, exist_ok=True)
    for pageno in pagenos(df):
        pagedf = get_page(df, pageno)
        pagefeats = trans.transform(pagedf)
        try:
            gen_blobs_for_page(pageno=pageno,
                               df=pagedf,
                               feats=pagefeats,
                               parentdf=df,
                               blobs_root=blobs_root,
                               chunkmgr=chunkmgr)
        except:
            dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
            LOG.debug("Exception thrown, removing {}", dirname)
            shutil.rmtree(dirname, ignore_errors=True)
            raise
