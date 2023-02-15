import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from IPython.display import display, Markdown
import acquire


#################################################### Separate Columns Function ####################################################


def separate_column_type_list(df):
    '''
        Creates 2 lists separating continous & discrete
        variables.
        
        Parameters
        ----------
        df : Pandas DataFrame
            The DataFrame from which columns will be sorted.
        
        Returns
        ----------
        continuous_columns : list
            Columns in DataFrame with numerical values.
        discrete_columns : list
            Columns in DataFrame with categorical values.
    '''
    continuous_columns = []
    discrete_columns = []
    
    for column in df.columns:
        if (df[column].dtype == 'int' or df[column].dtype == 'float') and ('id' not in column) and (df[column].nunique()>10):
            continuous_columns.append(column)
        elif(df[column].dtype == 'int' or df[column].dtype == 'float') and (df[column].nunique()>11):
            continuous_columns.append(column)
        else:
            discrete_columns.append(column)
            
    return continuous_columns, discrete_columns


#################################################### Stats Test Functions ####################################################


def eval_results_2(p, alpha, group1, group2):
    '''
        Test Hypothesis  using Statistics Test Output.
        This function will take in the p-value, alpha, and a name for the 2 variables
        you are comparing (group1 and group2) and return a string stating 
        whether or not there exists a relationship between the 2 groups. 
    '''
    if p < alpha:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Reject $H_0$"))
        display(Markdown( f'There exists some relationship between {group1} and {group2}. (p-value: {p:.4f})'))
    
    else:
        display(Markdown(f"### Results:"))
        display(Markdown(f"### Failed to Reject $H_0$"))
        display(Markdown( f'There is not a significant relationship between {group1} and {group2}. (p-value: {p:.4f})'))

def question_hypothesis_test(question_number,df,column_name,question,target,alpha=.05):
    num, cat = separate_column_type_list(df)
    
    if (target in cat) and (column_name in num):
        # calculation
        overall_fraud_mean = df[column_name].mean()
        fraud_sample = df[df[target] >= 7][target]
        t, p = stats.ttest_1samp(fraud_sample, overall_fraud_mean)
        value = t
        p_value = p/2
        
        # Output variables
        test = "1-Sample T-Test"

        # Markdown Formatting
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"# {question}"))
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no relationship between `{column_name}` and `{target}`"))
        display(Markdown(f"$H_A$: There is a relationship between `{column_name}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        eval_results_2(p_value, alpha, column_name, target)
        
    elif (target in cat) and (column_name in cat):
        # calculations
        observed = pd.crosstab(df[column_name], df[target])
        chi2, p, degf, expected = stats.chi2_contingency(observed)
        value = chi2
        p_value = p
        
        # Output variables
        test = "Chi-Square"

        # Markdown Formatting
        display(Markdown(f"# Question #{question_number}:"))
        display(Markdown(f"# {question}"))
        display(Markdown(f"### Hypothesis:"))
        display(Markdown(f"$H_0$: There is no relationship between `{column_name}` to `{target}`"))
        display(Markdown(f"$H_A$: There is a relationship between `{column_name}` and `{target}` "))
        display(Markdown(f"### Statistics Test:"))
        display(Markdown(f"### `{test} = {value}`"))

        eval_results_2(p_value, alpha, column_name, target)
    else:
        print("write code for different test")


#################################################### Visual Functions ####################################################

def big_question(df):
    sns.countplot(data = df, x ='fraud', palette='winter')
    sns.set_theme(style="whitegrid")
    plt.xlabel('Fraud?')
    plt.ylabel('Purchase count')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.title('How much Fraud is going on?')
    plt.show()


def question_1_visual(df):
    sns.countplot(data = df, x ='online_order', hue='fraud')
    sns.set_theme(style="whitegrid")
    plt.xlabel('Online Order')
    plt.ylabel('Purchase count')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.title('Do Fraud and Online Ordering have a Relationship?')
    title = 'Fraud'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()


def question_2_visual(df):
    sns.countplot(data = df, x ='used_pin_number', hue='fraud')
    sns.set_theme(style="whitegrid")
    plt.xlabel('Used Pin Number')
    plt.ylabel('Purchase Count')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.title('Does The Use Of a Pin and Fraud Have a Relationship?')
    title = 'Fraud'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()


def question_3_visual(df):
    question = "Is there a relationship between Citric Acid and Quality?"

    x = df['quality']
    y = df['citric_acid']

    fig, ax = plt.subplots()

    ax.bar(x,y, width=0.1, color="pink", zorder=0)
    sns.regplot(x=x, y=y, ax=ax)
    ax.set_ylim(0, None)
    plt.suptitle(f"{question}")

    plt.show()


def question_4_visual(df):
    question = "Is there a realationship between Free Sulfur Dioxide and Quality?"
    x = df["quality"]
    y = df["free_sulfur_dioxide"]

    fig, ax = plt.subplots()

    ax.bar(x,y, width=0.5, color="deepskyblue", zorder=0)
    sns.regplot(x=x, y=y, ax=ax,color="palevioletred")
    ax.set_ylim(0, None)
    plt.suptitle(f"{question}")
    plt.show()

