import json
import argparse
import sys
import pyserini as ps
from pyserini_pisa import PisaIndex, PisaRetrieve, PisaScorer, PisaStopwords, PISA_INDEX_DEFAULTS


def main():
  if not pt.started():
    pt.init()
  parser = argparse.ArgumentParser('pyserini_pisa')
  parser.set_defaults(func=lambda x: parser.print_help())
  subparsers = parser.add_subparsers()
  index_parser = subparsers.add_parser('index')
  index_parser.add_argument('index_path')
  index_parser.add_argument('dataset')
  index_parser.add_argument('--threads', type=int, default=8)
  index_parser.add_argument('--batch_size', type=int, default=100_000)
  index_parser.add_argument('--fields', nargs='+')
  index_parser.set_defaults(func=main_index)
  
  # retrieve_parser = subparsers.add_parser('retrieve')
  # retrieve_parser.add_argument('index_path')
  # retrieve_parser.add_argument('dataset')
  # retrieve_parser.add_argument('scorer', choices=PisaScorer.__members__.values(), type=PisaScorer)
  # retrieve_parser.add_argument('--num_results', '-k', type=int, default=1000)
  # retrieve_parser.add_argument('--stops', choices=PisaStopwords.__members__.values(), type=PisaStopwords, default=PISA_INDEX_DEFAULTS['stops'])
  # retrieve_parser.add_argument('--field', default=None)
  # retrieve_parser.add_argument('--threads', type=int, default=8)
  # retrieve_parser.add_argument('--batch_size', type=int, default=None)
  # retrieve_parser.set_defaults(func=main_retrieve)

  args = parser.parse_args()
  args.func(args)


def main_index(args):
  dataset = pt.get_dataset(args.dataset)
  index = PisaIndex(args.index_path, args.fields, threads=args.threads, batch_size=args.batch_size)
  docs = dataset.get_corpus_iter(verbose=False)
  total = None
  if hasattr(dataset, 'irds_ref'):
    total = dataset.irds_ref().docs_count()
  docs = pt.tqdm(docs, total=total, smoothing=0., desc='document feed', unit='doc')
  index.index(docs)

def main_index(args):
  #dataset = pt.get_dataset(args.dataset)
  
  #converts JSON collection (Pyserini) to plaintext (PISA)
  plaintext_path=os.path.join(os.path.dirname(args.collection), "out.txt")
  plaintext_collection = parse_json_collection_to_plaintext(args.collection,plaintext_path) # where to pass plaintext collection, where to store it?
  # modify args.fields to not include the input collection param?
  index = PisaIndex(args.index_path, args.fields, input=plaintext_path, threads=args.threads)

  # Old way:
  # docs = dataset.get_corpus_iter(verbose=False)
  # total = None
  # if hasattr(dataset, 'irds_ref'):
  #   total = dataset.irds_ref().docs_count()
  # docs = pt.tqdm(docs, total=total, smoothing=0., desc='document feed', unit='doc')
  # index.index(docs)

  # New way:
  index.index(plaintext_collection)

def parse_json_collection_to_plaintext(file_path, out_file):
  try:
      with open(file_path, 'r') as file, , open(out_file, 'w') as outfile:
        for line in file:
            data = json.loads(line)
            doc_id = data.get('id', '')
            contents = data.get('contents', '')
            outfile.write(f"{doc_id}   {contents}\n\n")
  except FileNotFoundError:
      print(f"File not found: {file_path}")
      return None
  except json.JSONDecodeError as e:
      print(f"Error decoding JSON: {e}")
      return None


def main_retrieve(args):
  #TODO: merge with Palvi/Leyang