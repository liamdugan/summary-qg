import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
sns.set_theme(style="whitegrid")
sns.set_palette(sns.color_palette("muted"))

df = pd.read_csv('data/questions/annotations.csv')

def get_majority(df):
  majority_df = df.drop(df.columns[5:], axis=1)
  majority_df['A'] = df[['A1', 'A2', 'A3']].mode(axis=1)
  return majority_df

print("-------")
print("RUNNING ANNOTATION ANALYSIS")
print("-------\n")

"""
Individual Annotator %ages and Pairwise IAA (Table 3)
"""
def analyze_per_annotator(df):
  def get_annotator_scores(df):
    a1 = df['A1'].value_counts(normalize=True)['Y']
    a2 = df['A2'].value_counts(normalize=True)['Y']
    a3 = df['A3'].value_counts(normalize=True)['Y']
    return (a1, a2, a3)

  print("Annotator scores for Acceptable: " + str(get_annotator_scores(df[df['Category'] == 'Acceptable'])))
  print("Annotator scores for Grammatical: " + str(get_annotator_scores(df[df['Category'] == 'Grammatical'])))
  print("Annotator scores for Interpretable: " + str(get_annotator_scores(df[df['Category'] == 'Interpretable'])))
  print("Annotator scores for Relevant: " + str(get_annotator_scores(df[df['Category'] == 'Relevant'])))
  print("Annotator scores for Correct: " + str(get_annotator_scores(df[df['Category'] == 'Correct'])))

def analyze_iaa(df):
  def get_pairwise_iaa(df):
    a1_a2 = cohen_kappa_score(df['A1'].tolist(), df['A2'].tolist())
    a2_a3 = cohen_kappa_score(df['A2'].tolist(), df['A3'].tolist())
    a3_a1 = cohen_kappa_score(df['A3'].tolist(), df['A1'].tolist())
    return (a1_a2, a2_a3, a3_a1)

  print("Pairwise IAA for Acceptable:" + str(get_pairwise_iaa(df[df['Category'] == 'Acceptable'])))
  print("Pairwise IAA for Grammatical:" + str(get_pairwise_iaa(df[df['Category'] == 'Grammatical'])))
  print("Pairwise IAA for Interpretable:" + str(get_pairwise_iaa(df[df['Category'] == 'Interpretable'])))
  print("Pairwise IAA for Relevant:" + str(get_pairwise_iaa(df[df['Category'] == 'Relevant'])))
  print("Pairwise IAA for Correct:" + str(get_pairwise_iaa(df[df['Category'] == 'Correct'])))

print("-------")
print("Individual Annotator %ages and Pairwise IAA (Table 3)")
print("-------")
analyze_per_annotator(df)
analyze_iaa(df)
print("\n")
"""
Comparison Across Chapters (Table 5)
"""
def analyze_per_chapter(df):
  chapter2 = ['2.0', '2.1.1', '2.1.2', '2.1.3', '2.1.4', '2.1.5', '2.1.6', '2.1.7', '2.2', '2.3', '2.4.0', '2.4.1', '2.4.2', '2.4.3', '2.4.4', '2.4.5', '2.5.0', '2.5.1']
  chapter3 = ['3.0', '3.1', '3.2.0', '3.2.1', '3.3.0', '3.3.1', '3.4.0', '3.4.1', '3.4.2', '3.4.3', '3.5', '3.6', '3.7']
  chapter4 = ['4.0', '4.1', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7.0', '4.7.1', '4.8', '4.9.0', '4.9.1', '4.10']

  def get_chapter_scores(df):
    ch2 = df[(df.Section.isin(chapter2))]['A'].value_counts(normalize=True)['Y']
    ch3 = df[(df.Section.isin(chapter3))]['A'].value_counts(normalize=True)['Y']
    ch4 = df[(df.Section.isin(chapter4))]['A'].value_counts(normalize=True)['Y']
    return (ch2, ch3, ch4)

  print("Per-Chapter scores for Acceptable: " + str(get_chapter_scores(df[df['Category'] == 'Acceptable'])))
  print("Per-Chapter scores for Grammatical: " + str(get_chapter_scores(df[df['Category'] == 'Grammatical'])))
  print("Per-Chapter scores for Interpretable: " + str(get_chapter_scores(df[df['Category'] == 'Interpretable'])))
  print("Per-Chapter scores for Relevant: " + str(get_chapter_scores(df[df['Category'] == 'Relevant'])))
  print("Per-Chapter scores for Correct: " + str(get_chapter_scores(df[df['Category'] == 'Correct'])))

print("-------")
print("Comparison Across Chapters (Table 5)")
print("-------")
analyze_per_chapter(get_majority(df))
print("\n")

"""
Barplot of Majority Evaluation Scores (Figure 3)
"""
def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.0f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.1f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

majority = get_majority(df).replace({"Y": 100, "N": 0})
sns.set(font_scale=1)
sns.set_style("whitegrid")
sns.set_context("paper")
p = sns.barplot(x="Category", y="A", hue="Author", hue_order=['Original Text', 'Automatic Summary', 'Human Summary'], data=majority, ci=None)
p.set_title("")
p.set_xlabel("")
p.set_ylabel("")
p.legend(loc='lower right')
show_values(p)
p.get_figure().savefig('summaries.pdf', bbox_inches='tight')
