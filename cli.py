import json

def main_index(args):
  #dataset = pt.get_dataset(args.dataset)
  
  #converts JSON collection (Pyserini) to plaintext (PISA)
  plaintext_collection = parse_json_collection_to_plaintext(args.collection,???) # where to pass plaintext collection, where to store it?
  
  index = PisaIndex(args.index_path, args.fields, threads=args.threads)

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
