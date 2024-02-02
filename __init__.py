import pyserini as ps 
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
from .indexers import PisaIndexer, PisaToksIndexer, PisaIndexingMode
from pyserini.search import LuceneSearcher,SimpleSearcher
from pyserini.index.lucene import LuceneIndexer
from pyserini.search.lucene import LuceneSearcher
from enum import Enum
from collections import Counter
import functools
import ir_datasets
import ctypes
from . import _pisathon


class PisaStemmer(Enum):
  """
  Represents a built-in stemming function from PISA
  """
  none = 'none'
  porter2 = 'porter2'
  krovetz = 'krovetz'

class PisaScorer(Enum):
  """
  Represents a built-in scoring function from PISA
  """
  bm25 = 'bm25'
  dph = 'dph'
  pl2 = 'pl2'
  qld = 'qld'
  quantized = 'quantized'

class PisaIndexEncoding(Enum):
  """
  Represents a built-in index encoding type from PISA.
  """
  ef = 'ef'
  single = 'single'
  pefuniform = 'pefuniform'
  pefopt = 'pefopt'
  block_optpfor = 'block_optpfor'
  block_varintg8iu = 'block_varintg8iu'
  block_streamvbyte = 'block_streamvbyte'
  block_maskedvbyte = 'block_maskedvbyte'
  block_interpolative = 'block_interpolative'
  block_qmx = 'block_qmx'
  block_varintgb = 'block_varintgb'
  block_simple8b = 'block_simple8b'
  block_simple16 = 'block_simple16'
  block_simdbp = 'block_simdbp'

class PisaQueryAlgorithm(Enum):
  """
  Represents a built-in query algorithm
  """
  wand = 'wand'
  block_max_wand = 'block_max_wand'
  block_max_maxscore = 'block_max_maxscore'
  block_max_ranked_and = 'block_max_ranked_and'
  ranked_and = 'ranked_and'
  ranked_or = 'ranked_or'
  maxscore = 'maxscore'


class PisaStopwords(Enum):
  """
  Represents which set of stopwords to use during retrieval
  """
  terrier = 'terrier'
  none = 'none'


PISA_INDEX_DEFAULTS = {
  'stemmer': PisaStemmer.porter2,
  'index_encoding': PisaIndexEncoding.block_simdbp,
  'query_algorithm': PisaQueryAlgorithm.block_max_wand,
  'stops': PisaStopwords.terrier,
}


class PisaIndex(LuceneIndexer): # find all places where pt.Indexer is called and replace with ps.LuceneIndexer (equivalent)
  def __init__ (self,
      index_dir: str,
      append: bool = False, # "Append documents."
      threads: int = 8,
      # args
      generatorClass: str = "DefaultLuceneDocumentGenerator", # "Document generator class in package 'io.anserini.index.generator'."
      fields: str = [], # "List of fields to index (space separated), in addition to the default 'contents' field."
      storePositions: bool = False, # "Boolean switch to index store term positions; needed for phrase queries."
      storeDocVectors: bool = False, # "Boolean switch to store document vectors; needed for (pseudo) relevance feedback."
      storeContents: bool = False, # "Boolean switch to store document contents."
      storeRaw: bool = False, # "Boolean switch to store raw source documents."
      keepStopwords: bool = False, # "Boolean switch to keep stopwords."
      stopwords: str = "", # "Path to file with stopwords."
      stemmer: str = "porter", # "Stemmer: one of the following {porter, krovetz, none}; defaults to 'porter'."
      whitelist: str = None, # "File containing list of docids, one per line; only these docids will be indexed."
      impact: bool = False, # "Boolean switch to store impacts (no norms)."
      # bm25.accurate: bool = False, # "Boolean switch to use AccurateBM25Similarity (computes accurate document lengths)."
      language: str = "en", # "Analyzer language (ISO 3166 two-letter code)."
      pretokenized: bool = False, # "index pre-tokenized collections without any additional stemming, stopword processing"
      analyzeWithHuggingFaceTokenizer: str = "", # "index a collection by tokenizing text with pretrained huggingface tokenizers"
      useCompositeAnalyzer: bool = False, # "index a collection using a Lucene Analyzer & a pretrained HuggingFace tokenizer")
      useAutoCompositeAnalyzer: bool = False, # "index a collection using the AutoCompositeAnalyzer"
      batch_size: int = 100_000, # allegedly Pyserini sypports batch indexing but idk
  ):
    self.index_dir = index_dir
    ppath = Path(index_dir)
    # before: Optional[Union[PisaStemmer, str]] = None
    if stemmer is not None: stemmer = PisaStemmer(stemmer) # after: stemmer_from_name(stemmer)
    #index_encoding: Optional[Union[PisaIndexEncoding, str]] = None, ?
    if index_encoding is not None: index_encoding = PisaIndexEncoding(index_encoding)
    # before: stops: Optional[Union[PisaStopwords, List[str]]] = None,
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
    self.stemmer = stemmer
    self.index_encoding = index_encoding
    self.batch_size = batch_size
    self.threads = threads
    #self.text_field = text_field
    #self.overwrite = overwrite
    self.stops = stops
    self.generatorClass = generatorClass
    self.fields = fields
    self.storePositions = storePositions
    self.storeDocVectors = storeDocVectors
    self.storeContents = storeContents
    self.storeRaw = storeRaw
    self.keepStopwords = keepStopwords
    self.stopwords = stopwords
    self.stemmer = stemmer
    self.whitelist = whitelist
    self.impact = impact
    self.bm25_accurate = bm25_accurate
    self.language = language
    self.pretokenized = pretokenized
    self.analyzeWithHuggingFaceTokenizer = analyzeWithHuggingFaceTokenizer
    self.useCompositeAnalyzer = useCompositeAnalyzer
    self.useAutoCompositeAnalyzer = useAutoCompositeAnalyzer
    self.batch_size = batch_size

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
  

class PisaRetrieve(LuceneSearcher):
  # scorer: Union[PisaScorer, str],
  def __init__(
    self, 
    index: Union[PisaIndex, str], 
    scorer = None, 
    num_results: int = 1000, 
    threads=None, 
    verbose=False, 
    stops=None, 
    query_algorithm=None, 
    query_weighted=None, 
    toks_scale=100., 
    **retr_args): # what is retr_args? 
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
      stops = self.index.stops
    self.stops = PisaStopwords(stops)
    if query_algorithm is None:
      query_algorithm = PISA_INDEX_DEFAULTS['query_algorithm']
    self.query_algorithm = PisaQueryAlgorithm(query_algorithm)
    if query_weighted is None:
      self.query_weighted = self.scorer == PisaScorer.quantized
    else:
      self.query_weighted = query_weighted
    self.toks_scale = toks_scale
    # _pisathon.prepare_index(self.index.path, encoding=self.index.index_encoding.value, scorer_name=self.scorer.value, **retr_args)
