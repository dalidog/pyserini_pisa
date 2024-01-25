class PisaIndex(pt.Indexer):
  def __init__(self,
      path: str,
      text_field: str = None,
      stemmer: Optional[Union[PisaStemmer, str]] = None,
      index_encoding: Optional[Union[PisaIndexEncoding, str]] = None,
      batch_size: int = 100_000,
      stops: Optional[Union[PisaStopwords, List[str]]] = None,
      threads: int = 8,
      overwrite=False):
    self.path = path
    ppath = Path(path)
    if stemmer is not None: stemmer = PisaStemmer(stemmer)
    if index_encoding is not None: index_encoding = PisaIndexEncoding(index_encoding)
    if stops is not None and not isinstance(stops, list): stops = PisaStopwords(stops)
    if (ppath/'pt_pisa_config.json').exists():
      with (ppath/'pt_pisa_config.json').open('rt') as fin:
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
    self.overwrite = overwrite
    self.stops = stops

  def transform(self, *args, **kwargs):
    raise RuntimeError(f'You cannot use {self} itself as a transformer. Did you mean to call a ranking function like .bm25()?')

  def built(self):
    return (Path(self.path)/'pt_pisa_config.json').exists()

  def index(self, it):
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

  @staticmethod
  def from_dataset(dataset: Union[str, Dataset], variant: str = 'pisa_porter2', version: str = 'latest', **kwargs):
    from pyterrier.batchretrieve import _from_dataset
    return _from_dataset(dataset, variant=variant, version=version, clz=PisaIndex, **kwargs)

  @staticmethod
  def from_ciff(ciff_file: str, index_path, overwrite: bool = False, stemmer = PISA_INDEX_DEFAULTS['stemmer']):
    import pyciff
    stemmer = PisaStemmer(stemmer)
    warn(f"Using stemmer {stemmer}, which may not match the stemmer used to construct {ciff_file}. You may need to instead pass stemmer='none' and perform the stemming in a pipeline to match the behaviour.")
    if os.path.exists(index_path) and not overwrite:
      raise FileExistsError(f'An index already exists at {index_path}')
    ppath = Path(index_path)
    ppath.mkdir(parents=True, exist_ok=True)
    pyciff.ciff_to_pisa(ciff_file, str(ppath/'inv'))
    # Move the files around a bit to where they are typically located
    (ppath/'inv.documents').rename(ppath/'fwd.documents')
    (ppath/'inv.terms').rename(ppath/'fwd.terms')
    # The current version of pyciff does not create a termlex file, but a future version might
    if (ppath/'inv.termlex').exists():
      (ppath/'inv.termlex').rename(ppath/'fwd.termlex')
    else:
      # If it wasn't created, create one from the terms file
      _pisathon.build_binlex(str(ppath/'fwd.terms'), str(ppath/'fwd.termlex'))
    # The current version of pyciff does not create a doclex file, but a future version might
    if (ppath/'inv.doclex').exists():
      (ppath/'inv.doclex').rename(ppath/'fwd.doclex')
    else:
      # If it wasn't created, create one from the documents file
      _pisathon.build_binlex(str(ppath/'fwd.documents'), str(ppath/'fwd.doclex'))
    with open(ppath/'pt_pisa_config.json', 'wt') as fout:
      json.dump({
        'stemmer': stemmer.value,
      }, fout)
    return PisaIndex(index_path, stemmer=stemmer)

  def to_ciff(self, ciff_file: str, description: str = 'from pyterrier_pisa'):
    assert self.built()
    import pyciff
    pyciff.pisa_to_ciff(str(Path(self.path)/'inv'), str(Path(self.path)/'fwd.terms'), str(Path(self.path)/'fwd.documents'), ciff_file, description)

  def get_corpus_iter(self, field='toks', verbose=True):
    assert self.built()
    ppath = Path(self.path)
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
    return PisaIndexer(self.path, text_field or self.text_field or 'text', mode, stemmer=self.stemmer, threads=threads or self.threads, batch_size=batch_size or self.batch_size)

  def toks_indexer(self, text_field=None, mode=PisaIndexingMode.create, threads=None, batch_size=None, scale=100.):
    if PisaStemmer(self.stemmer) != PisaStemmer.none:
      raise ValueError("To index from dicts, you must set stemmer='none'")
    return PisaToksIndexer(self.path, text_field or self.text_field or 'toks', mode, threads=threads or self.threads, batch_size=self.batch_size, scale=scale)