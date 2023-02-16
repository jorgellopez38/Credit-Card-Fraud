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
    sns.set_theme(style="whitegrid")
    sns.countplot(data = df, x ='fraud', palette='colorblind')
    plt.xlabel('Fraud?')
    plt.ylabel('Purchase count')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.title('How much Fraud is going on?')
    plt.show()


def question_1_visual(df):
    sns.countplot(data = df, x ='online_order', hue='fraud')
    sns.set_theme(style="whitegrid")
    plt.ylabel('Purchase count')
    plt.xlabel('.')
    plt.xticks(np.arange(2), ['No Online Order', 'Online Order'])
    plt.title('Do Fraud and Online Ordering have a Relationship?')
    title = 'Fraud'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()


def question_2_visual(df):
    sns.countplot(data = df, x ='used_pin_number', hue='fraud')
    sns.set_theme(style="whitegrid")
    plt.ylabel('Purchase Count')
    plt.xlabel('.')
    plt.xticks(np.arange(2), ['No Pin', 'Pin'])
    plt.title('Does The Use Of a Pin and Fraud Have a Relationship?')
    title = 'Fraud'
    mylabels = ['No', 'Yes']
    plt.legend(title=title, labels=mylabels)
    plt.show()


def question_3_visual(df):
    sns.scatterplot(data=df, x='distance_from_home', y='distance_from_last_transaction', 
                hue='fraud')
    plt.title('Are Distances Related?')
    plt.ylabel('Distance From Last Transaction')
    plt.xlabel('Distance From Home')
    plt.show()
    

def question_4_visual(df):
    sns.barplot(data=df.sample(1000), x='fraud', y='ratio_to_median_purchase_price')
    plt.title('Fraud Vs Ratio To Median Purchase Price')
    plt.ylabel('Ratio To Median Purchase Price')
    plt.xlabel('Fraud')
    plt.xticks(np.arange(2), ['No', 'Yes'])
    plt.show()


def question_hypothesis_test1(df):    
    a = 0.05
    observed = pd.crosstab(df['online_order'], df['fraud'])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # if statement to return our results
    if p > a:
        print("We fail to reject null hypothesis")
    else:
        print("We reject the null hypothesis, there is a relationship")
        
    print(f"chi2: {chi2}")
    print(f"p:    {p}")
        
    
def question_hypothesis_test2(df):    
    a = 0.05
    observed = pd.crosstab(df['online_order'], df['fraud'])
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # if statement to return our results
    if p > a:
        print("We fail to reject null hypothesis")
    else:
        print("We reject the null hypothesis, there is a relationship")
        
    print(f"chi2: {chi2}")
    print(f"p:    {p}")
    

def question_hypothesis_test3(df):
    
    '''a function which takes in a train data set and calculates and returns a
       pearsons r test for bathrooms and tax value'''
    
    # setting alpha
    a = 0.05
    
    # performing a t test
    corr, p = stats.pearsonr(df.distance_from_home, df.distance_from_last_transaction)
    corr, p

    # if statement to return our results
    if p > a:
        print("We fail to reject null hypothesis")
    else:
        print("We reject the null hypothesis, there is correlation")
        
    print(f"correlation: {corr}")
    print(f"p:           {p}")
    
    
def question_hypothesis_test4(df):
    
    '''a function which takes in a train data set and calculates and returns a
       pearsons r test for bathrooms and tax value'''
    
    # setting alpha
    a = 0.05
    
    # performing a t test
    corr, p = stats.pearsonr(df.ratio_to_median_purchase_price, df.fraud)
    corr, p

    # if statement to return our results
    if p > a:
        print("We fail to reject null hypothesis")
    else:
        print("We reject the null hypothesis, there is correlation")
        
    print(f"correlation: {corr}")
    print(f"p:           {p}")
        