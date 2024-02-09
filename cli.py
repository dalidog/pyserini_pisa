import json
import argparse
import os
import sys
import pyserini as ps
from pyserini_pisa import PisaIndex, PisaRetrieve, PisaScorer, PisaStopwords, PISA_INDEX_DEFAULTS


def main():
  if not pt.started(): # wut
    pt.init()
  parser = argparse.ArgumentParser('pyserini_pisa')
  parser.set_defaults(func=lambda x: parser.print_help())
  subparsers = parser.add_subparsers()
  index_parser = subparsers.add_parser('index')
  index_parser.add_argument('--batch_size', type=int, default=100_000) # debatable if we should keep
  index_parser.add_argument('--fields', nargs='+')
  index_parser.add_argument('--append', action='store_true', help='Append documents.')
  index_parser.add_argument('-collection', metavar='[class]', required=True, help='Collection class in io.anserini.collection.')
  index_parser.add_argument('--input', metavar='[path]', required=True, help='Input collection.')
  index_parser.add_argument('--index', metavar='[path]', required=True, help='Index path.')
  index_parser.add_argument('--uniqueDocid', action='store_true', help='Removes duplicate documents with the same docid during indexing.')
  index_parser.add_argument('--optimize', action='store_true', help='Optimizes index by merging into a single index segment.')
  index_parser.add_argument('--memoryBuffer', metavar='[mb]', type=int, default=4096, help='Memory buffer size in MB.')
  index_parser.add_argument('--threads', metavar='[num]', type=int, default=4, help='Number of indexing threads.')
  index_parser.add_argument('--verbose', action='store_true', help='Enables verbose logging for each indexing thread.')
  index_parser.add_argument('-quiet', action='store_true', help='Turns off all logging.')
  index_parser.add_argument('-options', action='store_true', help='Print information about options.')
  index_parser.add_argument('-shard.count', metavar='[n]', type=int, default=-1, help='Number of shards to partition the document collection into.')
  index_parser.add_argument('-shard.current', metavar='[n]', type=int, default=-1, help='The current shard number to generate (indexed from 0).')
  index_parser.add_argument('--generator', metavar='[class]', help='Document generator class in package "io.anserini.index.generator".')
  index_parser.add_argument('--fields', nargs='+', help='List of fields to index (space separated), in addition to the default "contents" field.')
  index_parser.add_argument('--storePositions', action='store_true', help='Boolean switch to index store term positions; needed for phrase queries.')
  index_parser.add_argument('--storeDocvectors', action='store_true', help='Boolean switch to store document vectors; needed for (pseudo) relevance feedback.')
  index_parser.add_argument('--storeRaw', action='store_true', help='Boolean switch to store raw source documents.')
  index_parser.add_argument('--keepStopwords', action='store_true', help='Boolean switch to keep stopwords.')
  index_parser.add_argument('--stopwords', metavar='[file]', help='Path to file with stopwords.')
  index_parser.add_argument('--stemmer', metavar='[stemmer]', help='Stemmer: one of the following {porter, krovetz, none}; defaults to "porter".')
  index_parser.add_argument('--whitelist', metavar='[file]', help='File containing list of docids, one per line; only these docids will be indexed.')
  index_parser.add_argument('--impact', action='store_true', help='Boolean switch to store impacts (no norms).')
  index_parser.add_argument('--bm25_accurate', action='store_true', help='Boolean switch to use AccurateBM25Similarity (computes accurate document lengths).')
  index_parser.add_argument('--language', metavar='[language]', help='Analyzer language (ISO 3166 two-letter code).')
  index_parser.add_argument('--pretokenized', action='store_true', help='index pre-tokenized collections without any additional stemming, stopword processing')
  index_parser.add_argument('--analyzeWithHuggingFaceTokenizer', help='index a collection by tokenizing text with pretrained huggingface tokenizers')
  index_parser.add_argument('--useCompositeAnalyzer', action='store_true', help='index a collection using a Lucene Analyzer & a pretrained HuggingFace tokenizer')
  index_parser.add_argument('--useAutoCompositeAnalyzer', action='store_true', help='index a collection using the AutoCompositeAnalyzer')
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
  #dataset = pt.get_dataset(args.dataset)
  
  #converts JSON collection (Pyserini) to plaintext (PISA)
  plaintext_path=os.path.join(args.collection, "out.txt")
  for filename in os.listdir(args.collection):
    if not (file_name.split('.')[-1].lower() == ".txt"):
      parse_json_collection_to_plaintext(args.collection,plaintext_path)
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
        # case where all in same file no array
        if file_name.split('.')[-1].lower() == ".jsonl":
          for line in file:
              data = json.loads(line)
              doc_id = data.get('id', '')
              contents = data.get('contents', '')
              outfile.write(f"{doc_id}   {contents}\n\n")
        else
          json_data = json.load(file)
          # case where 1 file and array of objects
          if isinstance(json_data, list):
                for item in json_data:
                  doc_id = item['id']
                  contents = item['contents']
                  outfile.write(f"{doc_id}   {contents}\n\n")
          else:
            # case where 1 file for each document
            doc_id = json_data['id']
            contents = json_data['contents']
            outfile.write(f"{doc_id}   {contents}\n\n")

  except FileNotFoundError:
      print(f"File not found: {file_path}")
      return None
  except json.JSONDecodeError as e:
      print(f"Error decoding JSON: {e}")
      return None


def main_retrieve(args):
  #TODO: merge with Palvi/Leyang