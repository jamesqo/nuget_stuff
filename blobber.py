import gc
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

def predict_for_part(df, feats, parentdf, chunkmgr):
    try:
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

        return magic.predict(feats, df)
    except MemoryError:
        # NOTE: To avoid lots of nested 'During handling ... another exception occurred' messages
        # in the stack trace, we need to exit the except block before recursing.
        pass

    LOG.debug("Encountered MemoryError, splitting DataFrame into 2 parts")
    split = m // 2
    assert split > 0

    df1, feats1 = df.iloc[:split], feats[:split]
    df2, feats2 = df.iloc[split:], feats[split:]

    # TODO: Ensure the gc is picking up on the fact that we're no longer using df/feats.
    gc.collect()

    recs1 = predict_for_part(df1, feats1, parentdf, chunkmgr)
    recs2 = predict_for_part(df2, feats2, parentdf, chunkmgr)
    recs1.update(recs2)
    return recs1

def gen_blobs_for_page(pageno, df, feats, parentdf, blobs_root, chunkmgr):
    dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
    LOG.debug("Generating blobs for page #{} in {}", pageno, dirname)

    recs_dict = predict_for_part(df, feats, parentdf, chunkmgr)

    os.makedirs(dirname, exist_ok=True)
    ids = list(df['id'])
    for id_ in ids:
        hexid = id_.encode('utf-8').hex()
        blob_fname = os.path.join(dirname, '{}.json'.format(hexid))
        if is_windows: # pylint: disable=W0125
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

    if args.force_refresh_blobs:
        shutil.rmtree(blobs_root, ignore_errors=True)
    os.makedirs(blobs_root, exist_ok=True)
    for pageno in pagenos(df):
        dirname = os.path.join(blobs_root, 'page{}'.format(pageno))
        if not args.force_refresh_blobs and os.path.isdir(dirname):
            LOG.debug("Blobs for page #{} already exist in {}, skipping", pageno, dirname)
            continue

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
            LOG.debug("Exception thrown, removing {}", dirname)
            shutil.rmtree(dirname, ignore_errors=True)
            raise
