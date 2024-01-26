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

#should we include pisastemmer
class PisaStemmer(Enum):
  """
  Represents a built-in stemming function from PISA
  """
  none = 'none'
  porter2 = 'porter2'
  krovetz = 'krovetz'



class PisaIndex(ps.LuceneIndexer): # find all places where pt.Indexer is called and replace with ps.LuceneIndexer (equivalent)
  def __init__(self,
      index_dir: str,
      text_field: str = None,
      stemmer: Optional[Union[PisaStemmer, str]] = None, #couldn't find it in the code but ChatGPT says pyserini supports stemming
      #index_encoding: Optional[Union[PisaIndexEncoding, str]] = None,
      #batch_size: int = 100_000,
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
    #self.batch_size = batch_size
    self.threads = threads
    self.overwrite = overwrite
    self.stops = stops

  def transform(self, *args, **kwargs):
    raise RuntimeError(f'You cannot use {self} itself as a transformer. Did you mean to call a ranking function like .bm25()?')

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
    return PisaIndexer(self.path, text_field or self.text_field or 'text', mode, stemmer=self.stemmer, threads=threads or self.threads)

  def toks_indexer(self, text_field=None, mode=PisaIndexingMode.create, threads=None, batch_size=None, scale=100.):
    if PisaStemmer(self.stemmer) != PisaStemmer.none:
      raise ValueError("To index from dicts, you must set stemmer='none'")
    return PisaToksIndexer(self.path, text_field or self.text_field or 'toks', mode, threads=threads or self.threads, scale=scale)
  

class PisaRetrieve(pt.Transformer):
  def __init__(self, index: Union[PisaIndex, str], scorer: Union[PisaScorer, str], num_results: int = 1000, threads=None, verbose=False, stops=None, query_algorithm=None, query_weighted=None, toks_scale=100., **retr_args):
    if isinstance(index, PisaIndex):
      self.index = index
    else:
      self.index = PisaIndex(index)
    assert self.index.built(), f"Index at {self.index.path} is not built. Before you can use it for retrieval, you need to index."
    self.scorer = PisaScorer(scorer)
    self.num_results = num_results
    self.retr_args = retr_args
    self.verbose = verbose
    self.threads = threads or self.index.threads
    if stops is None:
      stpps = self.index.stops
    self.stops = PisaStopwords(stops)
    if query_algorithm is None:
      query_algorithm = PISA_INDEX_DEFAULTS['query_algorithm']
    self.query_algorithm = PisaQueryAlgorithm(query_algorithm)
    if query_weighted is None:
      self.query_weighted = self.scorer == PisaScorer.quantized
    else:
      self.query_weighted = query_weighted
    self.toks_scale = toks_scale
    _pisathon.prepare_index(self.index.path, encoding=self.index.index_encoding.value, scorer_name=self.scorer.value, **retr_args)

  def transform(self, queries):
    assert 'qid' in queries.columns
    assert 'query' in queries.columns or 'query_toks' in queries.columns
    inp = []
    if 'query_toks' in queries.columns:
      pretok = True
      for i, toks_dict in enumerate(queries['query_toks']):
        if not isinstance(toks_dict, dict):
          raise TypeError("query_toks column should be a dictionary (qid %s)" % qid)
        toks_dict = {str(k): float(v * self.toks_scale) for k, v in toks_dict.items()} # force keys and str, vals as float, apply toks_scale
        inp.append((i, toks_dict))
    else:
      pretok = False
      inp.extend(enumerate(queries['query']))

    if self.verbose:
      inp = pt.tqdm(inp, unit='query', desc=f'PISA {self.scorer.value}')
    with tempfile.TemporaryDirectory() as d:
      shape = (len(queries) * self.num_results,)
      result_qidxs = np.ascontiguousarray(np.empty(shape, dtype=np.int32))
      result_docnos = np.ascontiguousarray(np.empty(shape, dtype=object))
      result_ranks = np.ascontiguousarray(np.empty(shape, dtype=np.int32))
      result_scores = np.ascontiguousarray(np.empty(shape, dtype=np.float32))
      size = _pisathon.retrieve(
        self.index.path,
        self.index.index_encoding.value,
        self.query_algorithm.value,
        self.scorer.value,
        '' if self.index.stemmer == PisaStemmer.none else self.index.stemmer.value,
        inp,
        k=self.num_results,
        threads=self.threads,
        stop_fname=self._stops_fname(d),
        query_weighted=1 if self.query_weighted else 0,
        pretokenised=pretok,
        result_qidxs=result_qidxs,
        result_docnos=result_docnos,
        result_ranks=result_ranks,
        result_scores=result_scores,
        **self.retr_args)
    result = queries.iloc[result_qidxs[:size]].reset_index(drop=True)
    result['docno'] = result_docnos[:size]
    result['score'] = result_scores[:size]
    result['rank'] = result_ranks[:size]
    return result

  def __repr__(self):
    return f'PisaRetrieve({repr(self.index)}, {repr(self.scorer)}, {repr(self.num_results)}, {repr(self.retr_args)}, {repr(self.stops)})'

  @staticmethod
  def from_dataset(dataset: Union[str, Dataset], variant: str = None, version: str = 'latest', **kwargs):
    from pyterrier.batchretrieve import _from_dataset
    return _from_dataset(dataset, variant=variant, version=version, clz=PisaRetrieve, **kwargs)

  def _stops_fname(self, d):
    if self.stops == PisaStopwords.none:
      return ''
    else:
      fifo = os.path.join(d, 'stops')
      stops = self.stops
      if stops == PisaStopwords.terrier:
        stops = _terrier_stops()
      with open(fifo, 'wt') as fout:
        for stop in stops:
          fout.write(f'{stop}\n')
      return fifo


@functools.lru_cache()
def _terrier_stops():
  Stopwords = pt.autoclass('org.terrier.terms.Stopwords')
  stops = list(Stopwords(None).stopWords)
  return stops


class DictTokeniser(pt.Transformer):
  def __init__(self, field='text', stemmer=None):
    super().__init__()
    self.field = field
    self.stemmer = stemmer or (lambda x: x)

  def transform(self, inp):
    from nltk import word_tokenize
    return inp.assign(**{f'{self.field}_toks': inp[self.field].map(lambda x: dict(Counter(self.stemmer(t) for t in word_tokenize(x.lower()) if t.isalnum() )))})


if __name__ == '__main__':
  from . import cli
  cli.main()

  