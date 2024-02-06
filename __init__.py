import pyserini as ps
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

class PisaStemmer(Enum):
  """
  Represents a built-in stemming function from PISA
  """
  none = 'none'
  porter2 = 'porter'
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


def log_level(on=True):
  _pisathon.log_level(1 if on else 0)

class PisaIndex(ps.LuceneIndexer): # find all places where pt.Indexer is called and replace with ps.LuceneIndexer (equivalent)
  def __init__(self,
      append: bool = False, # "Append documents."
      threads: int = 4,
      # abstract args
      collectionClass: str, # "Collection class in io.anserini.collection."
      input: str, # "Input collection."
      index: str, # "Index path."
      uniqueDocid: bool = False, # "Removes duplicate documents with the same docid during indexing."
      optimize: bool = False, # "Optimizes index by merging into a single index segment."
      memoryBuffer: int = 4096, # "Memory buffer size in MB."
      verbose: bool = False, # "Enables verbose logging for each indexing thread."
      quiet: bool = False, # "Turns off all logging."
      options: bool = False, # "Print information about options."
      shardCount: int = -1, # "Number of shards to partition the document collection into."
      shardCurrent: int = -1, # "The current shard number to generate (indexed from 0)."
      # args
      generatorClass: str = "DefaultLuceneDocumentGenerator", # "Document generator class in package 'io.anserini.index.generator'."
      fields: str = [] # "List of fields to index (space separated), in addition to the default 'contents' field."
      storePositions: bool = False, # "Boolean switch to index store term positions; needed for phrase queries."
      storeDocVectors: bool = False, # "Boolean switch to store document vectors; needed for (pseudo) relevance feedback."
      storeContents: bool = False, # "Boolean switch to store document contents."
      storeRaw: bool = False, # "Boolean switch to store raw source documents."
      keepStopwords: bool = False # "Boolean switch to keep stopwords."
      stopwords: str = "", # "Path to file with stopwords."
      stemmer: str = "porter", # "Stemmer: one of the following {porter, krovetz, none}; defaults to 'porter'."
      whitelist: str = None, # "File containing list of docids, one per line; only these docids will be indexed."
      impact: bool = False, # "Boolean switch to store impacts (no norms)."
      bm25.accurate: bool = False, # "Boolean switch to use AccurateBM25Similarity (computes accurate document lengths)."
      language: str = "en", # "Analyzer language (ISO 3166 two-letter code)."
      pretokenized: bool = False, # "index pre-tokenized collections without any additional stemming, stopword processing"
      analyzeWithHuggingFaceTokenizer: str = ""; # "index a collection by tokenizing text with pretrained huggingface tokenizers"
      useCompositeAnalyzer: bool = False, # "index a collection using a Lucene Analyzer & a pretrained HuggingFace tokenizer")
      useAutoCompositeAnalyzer: bool = False # "index a collection using the AutoCompositeAnalyzer"
      batch_size: int = 100_000, # allegedly Pyserini sypports batch indexing but idk

      #Pyterrier stuff probably?
      #overwrite=False,
      #text_field: str = None,
      ):
    self.index = index
    ppath = Path(index)
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
    self.collectionClass = collectionClass
    self.input = input
    self.uniqueDocid = uniqueDocid
    self.optimize = optimize
    self.memoryBuffer = memoryBuffer
    self.verbose = verbose
    self.quiet = quiet
    self.options = options
    self.shardCount = shardCount
    self.shardCurrent = shardCurrent



  def built(self):
    return (Path(self.index)/'ps_pisa_config.json').exists()

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
    ppath = Path(self.index)
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
    return PisaIndexer(self.index, text_field or self.text_field or 'text', mode, stemmer=self.stemmer, threads=threads or self.threads)

  def toks_indexer(self, text_field=None, mode=PisaIndexingMode.create, threads=None, batch_size=None, scale=100.):
    if PisaStemmer(self.stemmer) != PisaStemmer.none:
      raise ValueError("To index from dicts, you must set stemmer='none'")
    return PisaToksIndexer(self.index, text_field or self.text_field or 'toks', mode, threads=threads or self.threads, scale=scale)

class PisaRetrieve():
  #TODO: merge with Palvi/Leyang

#TODO: what is this?
@functools.lru_cache()
def _terrier_stops():
  Stopwords = pt.autoclass('org.terrier.terms.Stopwords')
  stops = list(Stopwords(None).stopWords)
  return stops

#TODO: change to Pyserini
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