import pyserini as ps #idk if this works bc im on my local machine
from pyserini.index import IndexWriter, SimpleIndexer
from pyserini.search import SimpleSearcher
import numpy as np
import json
import sys
from pathlib import Path
import tempfile
import os
import more_itertools
from warnings import warn
from typing import Optional, Union, List
from enum import Enum
from collections import Counter
import functools
import ir_datasets
from . import _pisathon
from .indexers import PisaIndexer, PisaToksIndexer, PisaIndexingMode

class PisaIndex(ps.LuceneIndexer): # find all places where pt.Indexer is called and replace with ps.LuceneIndexer (equivalent)
  def __init__(self,
      index_dir: str,
      text_field: str = None,
      stemmer: Optional[Union[PisaStemmer, str]] = None, #couldn't find it in the code but ChatGPT says pyserini supports stemming
      #index_encoding: Optional[Union[PisaIndexEncoding, str]] = None,
      batch_size: int = 100_000,
      #stops: Optional[Union[PisaStopwords, List[str]]] = None,
      threads: int = 8,
      #overwrite=False,
      storePositions = False,
      storeDocVectors = False,
      storeRaw = False):
    self.index_dir = index_dir
    ppath = Path(index_dir)
    if stemmer is not None: stemmer = PisaStemmer(stemmer)
    if index_encoding is not None: index_encoding = PisaIndexEncoding(index_encoding)
    if stops is not None and not isinstance(stops, list): stops = PisaStopwords(stops)
    if (ppath/'ps_pisa_config.json').exists(): #TODO:write ps_pisa_config.json
      with (ppath/'ps_pisa_config.json').open('rt') as fin:
        config = json.load(fin)
      if stemmer is None:
        stemmer = PisaStemmer(config['stemmer'])
      if stemmer.value != config['stemmer']:
        warn(f'requested stemmer={stemmer.value}, but index was constructed with {config["stemmer"]}')
    if stemmer is None: stemmer = PISA_INDEX_DEFAULTS['stemmer']
    if index_encoding is None: index_encoding = PISA_INDEX_DEFAULTS['index_encoding']
    if stops is None: stops = PISA_INDEX_DEFAULTS['stops']
    self.text_field = text_field
    self.stemmer = stemmer
    self.index_encoding = index_encoding
    self.batch_size = batch_size
    self.threads = threads
    #self.overwrite = overwrite
    self.stops = stops

  def built(self):
    return (Path(self.index_dir)/'ps_pisa_config.json').exists()

  def index(self, it):
    it = more_itertools.peekable(it)
    first_doc = it.peek()
    text_field = self.text_field
    if text_field is None: # infer the text field
      dict_field = [k for k, v in sorted(first_doc.items()) if k.endswith('toks') and isinstance(v, dict)]
      if dict_field:
        text_field = dict_field[0]
        warn(f'text_field not specified; using pre-tokenized field {repr(text_field)}')
      else:
        text_field = [k for k, v in sorted(first_doc.items()) if isinstance(v, str) and k != 'docno']
        assert len(text_field) >= 1, f"no str or toks fields found in document. Fields: {k: type(v) for k, v in first_doc.items()}"
        warn(f'text_field not specified; indexing all str fields: {text_field}')

    mode = PisaIndexingMode.overwrite if self.overwrite else PisaIndexingMode.create

    if isinstance(text_field, str) and isinstance(first_doc[text_field], dict):
      return self.toks_indexer(text_field, mode=mode).index(it)
    return self.indexer(text_field, mode=mode).index(it)

  def bm25(self, k1=0.9, b=0.4, num_results=1000, verbose=False, threads=None, query_algorithm=None, query_weighted=None, toks_scale=100.):
    return PisaRetrieve(self, scorer=PisaScorer.bm25, bm25_k1=k1, bm25_b=b, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm, query_weighted=query_weighted, toks_scale=toks_scale)

  def dph(self, num_results=1000, verbose=False, threads=None, query_algorithm=None, query_weighted=None, toks_scale=100.):
    return PisaRetrieve(self, scorer=PisaScorer.dph, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm, query_weighted=query_weighted, toks_scale=toks_scale)

  def pl2(self, c=1., num_results=1000, verbose=False, threads=None, query_algorithm=None, query_weighted=None, toks_scale=100.):
    return PisaRetrieve(self, scorer=PisaScorer.pl2, pl2_c=c, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm, query_weighted=query_weighted, toks_scale=toks_scale)

  def qld(self, mu=1000., num_results=1000, verbose=False, threads=None, query_algorithm=None, query_weighted=None, toks_scale=100.):
    return PisaRetrieve(self, scorer=PisaScorer.qld, qld_mu=mu, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm, query_weighted=query_weighted, toks_scale=toks_scale)

  def quantized(self, num_results=1000, verbose=False, threads=None, query_algorithm=None, query_weighted=None, toks_scale=100.):
    return PisaRetrieve(self, scorer=PisaScorer.quantized, num_results=num_results, verbose=verbose, threads=threads or self.threads, stops=self.stops, query_algorithm=query_algorithm, query_weighted=query_weighted, toks_scale=toks_scale)

  def num_terms(self):
    assert self.built()
    return _pisathon.num_terms(self.path)

  def num_docs(self):
    assert self.built()
    return _pisathon.num_docs(self.path)

  def __len__(self):
    return self.num_docs()

  def __repr__(self):
    return f'PisaIndex({repr(self.path)})'

# deleted to_ciff, from_ciff, from_dataset

  def get_corpus_iter(self, field='toks', verbose=True):
    assert self.built()
    ppath = Path(self.index_dir)
    assert (ppath/'fwd').exists(), "get_corpus_iter requires a fwd index"
    m = np.memmap(ppath/'fwd', mode='r', dtype=np.uint32)
    lexicon = [l.strip() for l in (ppath/'fwd.terms').open('rt')]
    idx = 2
    it = iter((ppath/'fwd.documents').open('rt'))
    if verbose:
      it = _logger.pbar(it, total=int(m[1]), desc=f'iterating documents in {self}', unit='doc')
    for did in it:
      start = idx + 1
      end = start + m[idx]
      yield {'docno': did.strip(), field: dict(Counter(lexicon[i] for i in m[start:end]))}
      idx = end

  def indexer(self, text_field=None, mode=PisaIndexingMode.create, threads=None, batch_size=None):
    return PisaIndexer(self.index_dir, text_field or self.text_field or 'text', mode, stemmer=self.stemmer, threads=threads or self.threads)

  def toks_indexer(self, text_field=None, mode=PisaIndexingMode.create, threads=None, batch_size=None, scale=100.):
    if PisaStemmer(self.stemmer) != PisaStemmer.none:
      raise ValueError("To index from dicts, you must set stemmer='none'")
    return PisaToksIndexer(self.index_dir, text_field or self.text_field or 'toks', mode, threads=threads or self.threads, scale=scale)