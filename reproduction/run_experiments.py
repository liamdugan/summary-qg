import argparse, torch, re, pandas, sys

sys.path.append('..')

from pipeline import pipeline
from transformers import pipeline as pipelineHF
from transformers import AutoTokenizer
from summary_qg import extract_qa_pairs

parser = argparse.ArgumentParser()
parser.add_argument('-s','--use_summary', help="Run automatic summarization rather than reading in automatic summary data from a file", action='store_true')
parser.add_argument('-f','--fast', help="Use the smaller and faster versions of the models", action='store_true')
args = parser.parse_args()

def extract_chapters(files):
    chapters = dict()
    for fname in files:
        text = open(fname,'r').read()
        for x in re.finditer("\n([2-4]\.[0-9]\.?[0-9]?)(.*?)\n(.*?)(\n|$)", text):
            chapters[x.group(1)] = x.group(3)
    return chapters

if __name__ == "__main__":
    original_text = extract_chapters(['../data/text/slp_ch2.txt', '../data/text/slp_ch3.txt', '../data/text/slp_ch4.txt'])
    human_summaries_A1 = extract_chapters(['../data/summaries/summary_A1.txt'])
    human_summaries_A2 = extract_chapters(['../data/summaries/summary_A2.txt'])
    human_summaries_A3 = extract_chapters(['../data/summaries/summary_A3.txt'])        
    auto_summaries = extract_chapters(['../data/summaries/summary_auto.txt']) if not args.use_summary else original_text

    sources = {
        'Original Text': original_text,
        'Auto Summary': auto_summaries,
        'Human Summary (A1)': human_summaries_A1,
        'Human Summary (A2)': human_summaries_A2,
        'Human Summary (A3)': human_summaries_A3
    }
    
    qg_model = "valhalla/t5-small-qa-qg-hl" if args.fast else "valhalla/t5-base-qa-qg-hl"
    sum_model = "sshleifer/distilbart-cnn-6-6" if args.fast else "facebook/bart-large-cnn"
    
    qg = pipeline("multitask-qa-qg", model=qg_model)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    if torch.cuda.is_available():
        summarizer = pipelineHF("summarization", model=sum_model, device=0) if args.use_summary else None
    else:
        summarizer = pipelineHF("summarization", model=sum_model) if args.use_summary else None

    data = []
    for source, text in sources.items():
        for chapter, chtext in text.items():
            if source == 'Auto Summary':
                qa_pairs = extract_qa_pairs(tokenizer, qg, summarizer, chtext)
            else:
                qa_pairs = extract_qa_pairs(tokenizer, qg, None, chtext)
                
            for pair in qa_pairs:
                data.append([source, chapter, pair['question'], pair['answer']])
                
    df = pandas.DataFrame(data, columns=['Author','Section','Question','Answer'])
    
    df.to_csv('out.csv', index=False)
