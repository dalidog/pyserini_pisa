def main_index(args):
  dataset = pt.get_dataset(args.dataset)
  index = PisaIndex(args.index_path, args.fields, threads=args.threads, batch_size=args.batch_size)
  docs = dataset.get_corpus_iter(verbose=False)
  total = None
  if hasattr(dataset, 'irds_ref'):
    total = dataset.irds_ref().docs_count()
  docs = pt.tqdm(docs, total=total, smoothing=0., desc='document feed', unit='doc')
  index.index(docs)