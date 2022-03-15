import argparse, torch
from pipeline import pipeline
from transformers import pipeline as pipelineHF
from transformers import AutoTokenizer
from summary_qg import extract_qa_pairs

parser = argparse.ArgumentParser()
parser.add_argument('-s','--use_summary', help="Include summarization pre-processing", action='store_true')
parser.add_argument('-f','--fast', help="Use the smaller and faster versions of the models", action='store_true')
parser.add_argument('-i','--infile', help="The name of the text file to generate questions from. \
                                           If no file is given questions, are generated on user input", type=str)
args = parser.parse_args()

def print_qa_pairs(qa_pairs):
    for pair in qa_pairs:
        print(pair)

if __name__ == "__main__":

    qg_model = "valhalla/t5-small-qa-qg-hl" if args.fast else "valhalla/t5-base-qa-qg-hl"
    sum_model = "sshleifer/distilbart-cnn-6-6" if args.fast else "facebook/bart-large-cnn"

    print("Loading QG Model...")
    qg = pipeline("multitask-qa-qg", model=qg_model)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")

    if args.use_summary: print("Loading Summarization Model...")

    if torch.cuda.is_available():
        summarizer = pipelineHF("summarization", model=sum_model, device=0) if args.use_summary else None
    else:
        summarizer = pipelineHF("summarization", model=sum_model) if args.use_summary else None

    if args.infile:
        input_text = open(args.infile,"r").read()
        qa_pairs = extract_qa_pairs(tokenizer, qg, summarizer, input_text)
        print_qa_pairs(qa_pairs)

    else:
        while True:
            input_text = input(">")
            qa_pairs = extract_qa_pairs(tokenizer, qg, summarizer, input_text)
            print_qa_pairs(qa_pairs)
