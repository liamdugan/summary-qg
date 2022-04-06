import argparse, itertools
from pipeline import pipeline
from nltk import sent_tokenize
from unidecode import unidecode
from unicodedata import normalize

'''
  This function takes in an input of arbitrary length text and computes the minimal amount of times
  we have to split this text in order to satisfy these two constraints
    1) Text is never split mid-sentence
    2) No individual split is larger than 512 tokens long
'''
def create_splits(text, tokenizer):
    sents = sent_tokenize(text)
    enc_sents_len = [len(tokenizer.encode(s)[:-1]) for s in sents]
    num_splits = 0
    epsilon = 0.00001

    while True:
        num_splits += 1
        splits = []
        i = 0.0
        
        # Distribute the sentences as equally as possible among the num_splits
        while i < len(sents) - epsilon:
            splits.append((int(i),int(i + (len(sents) / num_splits) + epsilon)))
            i += len(sents) / num_splits

        # If every split in this set is <512 tokens long, keep it. Otherwise keep looping
        splits_len = [sum(enc_sents_len[s:e]) + 1 for (s,e) in splits]
        if all(s < 512 for s in splits_len):
            break
        elif len(sents) == 1:
            print(f'WARNING: sentence too long ({len(enc_sents_len[0])} tokens), skipped')
            return []

    return [' '.join(sents[s:e]) for (s,e) in splits]

def extract_qa_pairs(tokenizer, qa, summarizer, text):
    text = unidecode(normalize("NFKC", text))
    qa_pairs = []

    if summarizer:
        text = ' '.join([summarizer(split)[0]['summary_text'].strip() for split in create_splits(text, tokenizer)])
        print('\nSummary: ' + text + '\n')

    for split in create_splits(text, tokenizer):
        qa_pairs.extend(qa(split))

    return qa_pairs
