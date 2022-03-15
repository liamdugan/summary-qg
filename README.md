# Joint Summarization & Question Generation
![/data/media/demo.gif](/data/media/demo.gif)

This repository contains the code for the ACL 2022 paper "A Feasibility Study of Answer-Unaware Question Generation for Education". In our paper we show that running QG on summarized text results in higher quality questions.

## Installation

Conda:
```
conda create -n sumqg_env python=3.9.7
conda activate sumqg_env
pip install -r requirements.txt
python -m nltk.downloader punkt
```
venv:
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## Usage

To run QG on user input or a file, use `run_qg.py`. Add the `-s` flag to include automatic summarization in the pipeline before running QG (for use on longer inputs only). Add the `-f` flag to use the smaller and faster distilled versions of the models. The full options are listed below.
```
$ python run_qg.py -h
  -s, --use_summary     Include summarization pre-processing
  -f, --fast            Use the smaller and faster versions of the models
  -i, --infile          The name of the text file to generate questions from.
                        If no file is given, questions are generated on user input
```

Example (User Input):
```
$ python run_qg.py
>The answer to life is 42. The answer to most other questions is unknowable.
{'answer': '42', 'question': 'What is the answer to life?'}
{'answer': 'unknowable', 'question': 'What is the answer to most other questions?'}
```

Example (File Input):
```
$ python run_qg.py -s -i data/text/slp_ch2.txt

Summary: The dialogue above is from ELIZA, an early natural language <...>

{'answer': 'Eliza', 'question': "Who's mimicry of human conversation was remarkably successful?"}
{'answer': 'restaurants', 'question': 'Modern conversational agents can answer questions, book flights, or find what?'}
{'answer': 'Regular expressions', 'question': 'What can be used to specify strings we might want to extract from a document?'}
...
```

These scripts will default to using GPU if it is available. It is highly recommended (but not required) to have access to a CUDA-capable GPU when running these models. They are quite large and take a long time to run on CPU.

## Reproduction

To reproduce the results from the paper, use `reproduction/run_experiments.py`. This script will generate a file named `out.csv` that contains questions from all three sources (Automatic Summary, Original Text, Human Summary) separated by chapter subsection. If using the full-size models, this should take about 5-10 minutes on GPU.
```
$ python run_experiments.py -h
  -s, --use_summary  Run automatic summarization rather than reading in
                     automatic summary data from a file
  -f, --fast         Use the smaller and faster versions of the models
```

For example, this command will run the full QG model on all sources
```
$ cd reproduction
$ python run_experiments.py -s
```

To reproduce the coverage analysis, use `reproduction/coverage.py`. This script will print out the % of bolded key-terms from the textbook present in question-answer pairs in a given input csv file separated by textual source.
```
$ python coverage.py <keyword_file> <data_file>
```

For example, this command will run a coverage analysis on the data included in the paper. You may also choose to set `data_file` to the `out.csv` file to verify the coverage of your generated questions.
```
$ python coverage.py ../data/keywords/keywords.csv ../data/questions/questions.csv
```

Finally, to reproduce our analysis of annotations collected, use `reproduction/analyze_annotations.py`. This script will print out pairwise IAA and per-annotator statistics (Table 3) for each annotation questions as well as a breakdown across chapters (Table 5). It will also output the plot used in Figure 3 as `summaries.pdf`.
```
$ python analyze_annotations.py
```

## Model Details

The QG models used and the inference code to run them come from [Suraj Patil's amazing question_generation repository](https://github.com/patil-suraj/question_generation). Many thanks to him for sharing his great work with the academic community. Please see our paper for more details about the training and model inference.

Below are the evaluation results for the `t5-base` and `t5-small` models on the SQuAD1.0 dev set. For decoding, beam search with num_beams 4 was used with max decoding length set to 32. The [nlg-eval](https://github.com/Maluuba/nlg-eval) package was used to calculate the metrics.

| Name                                                                       | BLEU-4  | METEOR  | ROUGE-L | QA-EM  | QA-F1  |
|----------------------------------------------------------------------------|---------|---------|---------|--------|--------|
| [t5-base-qa-qg-hl](https://huggingface.co/valhalla/t5-base-qa-qg-hl)       | 21.0141 | 26.9113 | 43.2484 | 82.46  | 90.272 |
| [t5-small-qa-qg-hl](https://huggingface.co/valhalla/t5-small-qa-qg-hl)     | 18.9872 | 25.2217 | 40.7893 | 76.121 | 84.904 |

<br/>Below are the evaluation results for the `bart-large` and `distilbart` models on the CNN/DailyMail test set.

| Name                                                                       | ROUGE-2  | ROUGE-L |
|----------------------------------------------------------------------------|---------|---------|
| [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)       | 21.06 | 30.63 |
| [sshleifer/distilbart-cnn-6-6](https://huggingface.co/sshleifer/distilbart-cnn-6-6)     | 20.17 | 29.70 |

## Citation
If you use our code or findings in your research, please cite us as:
```
@article{dugan2022feasibility,
  title={A Feasibility Study of Answer-Unaware Question Generation for Education},
  author={Dugan, Liam and Miltsakaki, Eleni and Upadhyay, Shriyash and Ginsberg, Etan and Gonzalez, Hannah and Choi, Dahyeon and Yuan, Chuning and Callison-Burch, Chris},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```