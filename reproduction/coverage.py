import pandas, argparse

parser = argparse.ArgumentParser(description='Do coverage analysis for keywords')
parser.add_argument('keyword_file', help='A file containing keywords', type=str)
parser.add_argument('data_file', help='A file containing questions and answers', type=str)
args = parser.parse_args()

keyword_df = pandas.read_csv(args.keyword_file)
keywords = keyword_df.words.unique()
data = pandas.read_csv(args.data_file)

'''
  Computes the percentage of keywords that were present as exact substrings in a given set of question-answer pairs.
  Output is a tuple of the form (% in questions, % in answers, % in both Qs and As)
'''
def percent_keywords_used_in_questions(input_df):
    def exact_match(series):
        return lambda keyword: series.apply(lambda q: keyword in q).any()

    def compute_coverage(keywords, data):
        fq = exact_match(data['Question'])
        fa = exact_match(data['Answer'])

        q_mask = keywords['words'].apply(fq)
        a_mask = keywords['words'].apply(fa)
        qa_mask = q_mask | a_mask

        return tuple(len(keywords[mask]) / len(keywords) for mask in [q_mask, a_mask, qa_mask])

    return compute_coverage(keyword_df, input_df)

for author in data.Author.unique():
    print(author, percent_keywords_used_in_questions(data[data['Author'] == author]))
    
print("Human Summary (Total)", percent_keywords_used_in_questions(data[data['Author'].str.contains("Human Summary")]))
