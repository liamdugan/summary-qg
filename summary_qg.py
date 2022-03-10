#!pip install -U transformers==3.0.0 --quiet
#!pip install unidecode
#!pip install nltk
#!pip install torch
#!python -m nltk.downloader punkt

import math, logging, argparse
from pipeline import pipeline
from transformers import pipeline as pipelineHF
from transformers import AutoTokenizer
from nltk import sent_tokenize
from unidecode import unidecode
from unicodedata import normalize

parser = argparse.ArgumentParser()
parser.add_argument('-s','--use_summary', help="Include summarization pre-processing", action='store_true')
args = parser.parse_args()

'''
  This function takes in an input of arbitrary length text and computes the minimal amount of times 
  we have to split this text in order to satisfy these two constraints
    1) Text is never split mid-sentence
    2) No individual split is larger than 512 tokens long

  This function is necessary in order for the QA-QG model to be applied to arbitrarily
  long sequences of text (as its max length is 512 tokens)
'''
def create_splits(text, tokenizer):
  logging.disable(logging.WARNING)
  sents = sent_tokenize(text)
  epsilon = 0.00001
  num_splits = 0

  # Loop until we have a valid set of splits
  while True:
    # Increment num_splits and calculate this loop's number of sentences per split
    num_splits += 1
    sents_per_split = len(sents) / num_splits

    # Loop through the sentences in the text and distribute them equally among the num_splits
    # (sents_per_split * num_splits = len(sents) +/- epsilon)
    splits = []
    i = 0.0
    while i < len(sents) - epsilon:
      splits.append((int(i),int(i + sents_per_split + epsilon)))
      i += sents_per_split

    # If every split in this set is less than 512 tokens long, keep it. Otherwise keep looping
    text_splits = [' '.join(sents[s:e]) for (s,e) in splits]
    if all(len(tokenizer.encode(s)) < 512 for s in text_splits):
      break
      
  return text_splits

def extract_qa_pairs(tokenizer, qa, summarizer, text):
  # Standardize text to avoid errors arising from weird characters
  text = unidecode(normalize("NFKC", text))
  qa_pairs = []

  if summarizer:
    # Pass the text through BART before feeding it into the QA-QG model
    text = ' '.join([summarizer(split)[0]['summary_text'].strip() for split in create_splits(text, tokenizer)])
    print('\nSummary: ' + text + '\n')
    
  for split in create_splits(text, tokenizer):
    # Pass each split into the QA-QG model and compile the outputs together
    qa_pairs.extend(qa(split))

  return qa_pairs


if __name__ == "__main__":

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  
  nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl", device=device) # (This uses T5-base fine-tuned for multitask question answering & generation)
  tokenizer = AutoTokenizer.from_pretrained("t5-base")

  if args.use_summary:
    summarizer = pipelineHF("summarization", model="facebook/bart-large-cnn", device=device) # (This uses BART-large fine-tuned on CNN/DailyMail for summarization)
  else:
    summarizer = None

  while True:
    input_text = input("Input Text:")
    qa_pairs = extract_qa_pairs(tokenizer, nlp, summarizer, input_text)
    print(qa_pairs)
