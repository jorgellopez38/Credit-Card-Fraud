import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix, accuracy_score, precision_score, recall_score
from pydataset import data
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from IPython.display import display, Markdown


#################################################### Split Function ####################################################


def split_data(df, target):
    '''
    This function take in a dataframe performs a train, validate, test split
    Returns train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test
    and prints out the shape of train, validate, test
    '''
    
    #create train_validate and test datasets
    train, test = train_test_split(df, train_size = 0.8, random_state = 123)
    #create train and validate datasets
    train, validate = train_test_split(train, train_size = 0.7, random_state = 123)

    #Split into X and y
    X_train = train.drop(columns=[target])
    y_train = train[target]

    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    X_test = test.drop(columns=[target])
    y_test = test[target]

    # Have function print datasets shape
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')
   
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test

#################################################### Baseline Function ###############################################

def get_baseline(df):
    '''This function generates our baseline and prints out in markdown'''
    
    df['baseline'] = df['fraud'].value_counts().idxmax()
    (df['fraud'] == df['baseline']).mean()
    # clean f string markdown
    display(Markdown(f"### Baseline: {(df['fraud'] == df['baseline']).mean()*100:.2f}%"))


#################################################### ML Functions ####################################################


def get_rf(X_train, y_train, X_validate, y_validate):   
    '''This function generates the best random forest model that we found and prints out the accuracy on train and validate'''
    
    rf = RandomForestClassifier(max_depth=5, random_state=42,
                            max_samples=0.5)
    #fit it 
    rf.fit(X_train, y_train)
    # clean f string
    print('Random Forest Model')
    print(f"Accuracy of Random Forest on train data: {rf.score(X_train, y_train)*100:.2f}%") 
    print(f"Accuracy of Random Forest on validate: {rf.score(X_validate, y_validate)*100:.2f}%")
    
    
def get_logit(X_train, y_train, X_validate, y_validate):
    '''This function generates the best logistic regression model that we found and prints out the accuracy on train and validate'''
    
    logit = LogisticRegression(C=.1, random_state=42, 
                           intercept_scaling=1, solver='newton-cg')

    #fit the model
    logit.fit(X_train, y_train, )
    #clean f string
    print('Logistic Regression Model')
    print(f"Accuracy of Logistic Regression on train: {logit.score(X_train, y_train)*100:.2f}%") 
    print(f"Accuracy of Logistic Regression on validate: {logit.score(X_validate, y_validate)*100:.2f}%")


def get_clf(X_train, y_train, X_validate, y_validate):
    '''This function generates the best decision tree model that we found and prints out the accuracy on train and validate'''
    
    clf = DecisionTreeClassifier(max_depth=5)

    #fit the model
    clf.fit(X_train, y_train)
    #clean f string
    print('Decision Tree Model')
    print(f"Accuracy of Decision Tree on train: {clf.score(X_train, y_train)*100:.2f}%") 
    print(f"Accuracy of Decision Tree on validate: {clf.score(X_validate, y_validate)*100:.2f}%")
    
        

def get_knn(X_train, y_train, X_validate, y_validate):
    '''This function generates the best k nearest neighbor model that we found and prints out the accuracy on train and validate'''
    
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X_train, y_train)
    knn.score(X_train, y_train)
    knn.score(X_validate, y_validate)
    # clean f string
    print('KNN Model')
    print(f"Accuracy of KNN on train: {knn.score(X_train, y_train)*100:.2f}%") 
    print(f"Accuracy of KNN on validate: {knn.score(X_validate, y_validate)*100:.2f}%")
    

#################################################### Top ML Models Function ####################################################


def get_top_models(X_train, y_train, X_validate, y_validate):
    '''This function gets all the top ML models and plots them together for a visual'''

    # best Random Forest
    best_rf = RandomForestClassifier(max_depth=5, random_state=42, max_samples=0.5)
    best_rf.fit(X_train, y_train)

    best_rf_train_score = best_rf.score(X_train, y_train)
    best_rf_validate_score = best_rf.score(X_validate, y_validate)

    # Best KNN
    best_knn = KNeighborsClassifier(n_neighbors=2)
    best_knn.fit(X_train, y_train)

    best_knn_train_score = best_knn.score(X_train, y_train)
    best_knn_validate = best_knn.score(X_validate, y_validate)

    # Best Logistic Regression
    best_lr = LogisticRegression(C=.1, random_state=42,intercept_scaling=1, solver='newton-cg')
    best_lr.fit(X_train, y_train)

    best_lr_train_score = best_lr.score(X_train, y_train)
    best_lr_validate_score = best_lr.score(X_validate, y_validate)

    # Best Decision Tree
    best_clf = DecisionTreeClassifier(max_depth=5)   
    best_clf.fit(X_train, y_train)

    best_clf_train_score = best_clf.score(X_train, y_train)
    best_clf_validate_score = best_clf.score(X_validate, y_validate)

    # lists with model names & score information
    best_model_name_list = ["KNN","Random_Forest","Logistic_Regression","Decision Tree"]
    best_model_train_scores_list = [best_knn_train_score,best_rf_train_score,best_lr_train_score,best_clf_train_score]
    best_model_validate_scores_list = [best_knn_validate,best_rf_validate_score,best_lr_validate_score,best_clf_validate_score]
    
    # new empty DataFrame
    best_scores_df = pd.DataFrame()

    # new columns using lists for data
    best_scores_df["Model"] = best_model_name_list
    best_scores_df["Train_Score"] = best_model_train_scores_list
    best_scores_df["Validate_Score"] = best_model_validate_scores_list

    # plot it
    plt.figure(figsize=(11, 8.5))
    ax = best_scores_df.plot.bar(rot=5)
    plt.xticks(np.arange(4), ['KNN', 'Random Forest','Logistic Regression', 'Decision Tree'])
    plt.ylabel('Scores')
    plt.title('Top Models')
    sns.set_theme(style="whitegrid")
    ax.annotate('Best Model',fontsize=12,color="Black",weight="bold", xy=(1, 1), 
                xytext=(.65, .9))
    mylabels = ['Train', 'Validate']
    ax.legend(labels=mylabels,bbox_to_anchor=(1.02, 1), loc='upper left',borderaxespad=0)
    plt.show()



#################################################### Test Functions ####################################################
    

def get_test(X_train, y_train, X_test, y_test):
    '''
    This function gets our best peforming model and runs it on our test data
    '''
    # random forest model was best
    rf = RandomForestClassifier(max_depth=5, random_state=42,
                            max_samples=0.5)
    rf.fit(X_train, y_train)
    rf.score(X_test,y_test)
    
     # clean f string
    display(Markdown(f'### Random Forest Model'))
    display(Markdown(f'### Accuracy on Test {rf.score(X_test,y_test)*100:.2f}%'))
    

def get_mvb(X_train, y_train, X_test, y_test, df):
    '''This function plots the test data and plot the baseline together for a final visual'''
    
    # Recalculating Best Peforming Model with new name
    best_model = RandomForestClassifier(max_depth=5, random_state=42,
                            max_samples=0.5)  
    best_model.fit(X_train, y_train)
    best_model.score(X_test,y_test)
    
    # Baseline
    plot_baseline = (df['fraud'] == df['baseline']).mean() 
    
    # Best Performing Model(Logistic Regression Combo{c=100,newton-cg}) Test Score: 
    best_test_score = best_model.score(X_test,y_test)  
    
    # Test Scores: Project Baseline vs Best Model
    plot_baseline, best_test_score
    
    # Temporary Dictionary Holding Baseline & Model Test Score
    best_model_plot={"Baseline":[plot_baseline], "Test":[best_test_score]}
    
    # Converting Temporary Dictionary to DataFrame
    best_model_plot = pd.DataFrame(best_model_plot)
    
    # Visualizing Both Baseline & Model Test Scores
    fig=sns.barplot(data= best_model_plot,palette="colorblind")
    plt.title("Best Model vs. Baseline")
    fig.set(ylabel='Scores')
    plt.show()




