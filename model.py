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
    


def recall_tree(X_train, y_train, X_validate, y_validate):
    """
    This function runs the Decision Tree classifier on the training and validation test sets for recall.
    """
    #Create the model
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    #Train the model
    clf = clf.fit(X_train, y_train)

    #Recall
    #Make a prediction from the model
    y_pred = clf.predict(X_train)
    y_pred_val = clf.predict(X_validate)

    train_score = recall_score(y_train, y_pred, average='micro')
    val_score = recall_score(y_validate, y_pred_val, average='micro')
    method = 'Recall'
        
    #Print the score
    print(f'{method} for Decision Tree classifier on training set:   {train_score:.4f}')
    print(f'{method} for Decision Tree classifier on validation set: {val_score:.4f}')
    print(classification_report(y_validate, y_pred_val))


#################################################### Combined Algorithm Functions ####################################################


def lr_mod(X_train, y_train, X_validate, y_validate, metric = 1, print_scores = False):
    """
    This function runs the Logistic Regression classifier on the training and validation test sets.
    """
    #Creating a logistic regression model
    logit = LogisticRegression(C=.1, random_state=42,intercept_scaling=1, solver='newton-cg')

    #Fitting the model to the train dataset
    logit.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_validate)        
        
        train_score = logit.score(X_train, y_train)
        val_score =  logit.score(X_validate, y_validate)
        method = 'Accuracy'

    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_validate)

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_validate, y_pred_val, average='micro')
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = logit.predict(X_train)
        y_pred_val = logit.predict(X_validate)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_validate, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Logistic Regression classifier on training set:   {train_score:.4f}')
        print(f'{method} for Logistic Regression classifier on validation set: {val_score:.4f}')
        print(classification_report(y_validate, y_pred_val))
    
    #return train_score, val_score



def rand_forest(X_train, y_train, X_validate, y_validate, metric = 1, print_scores = False):
    """
    This function runs the Random Forest classifier on the training and validation test sets.
    """
    #Creating the random forest object
    rf = RandomForestClassifier(max_depth=5, random_state=42, class_weight='balanced', 
                                n_estimators=100, min_samples_leaf=5)
    
    #Fit the model to the train data
    rf.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_validate)
        
        train_score = rf.score(X_train, y_train)
        val_score =  rf.score(X_validate, y_validate)
        method = 'Accuracy'
    
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_validate)

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_validate, y_pred_val, average='micro')
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = rf.predict(X_train)
        y_pred_val = rf.predict(X_validate)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_validate, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Random Forest classifier on training set:   {train_score:.4f}')
        print(f'{method} for Random Forest classifier on validation set: {val_score:.4f}')
        print(classification_report(y_validate, y_pred_val))

    #return train_score, val_score
    


def dec_tree(X_train, y_train, X_validate, y_validate, metric = 1, print_scores = False):
    """
    This function runs the Decision Tree classifier on the training and validation test sets.
    """
    #Create the model
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    
    #Train the model
    clf = clf.fit(X_train, y_train)
    
    #Accuracy
    if metric == 1:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_validate)
        
        train_score = clf.score(X_train, y_train)
        val_score =  clf.score(X_validate, y_validate)
        method = 'Accuracy'
    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_validate)

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_validate, y_pred_val, average='micro')
        method = 'Precision'
        
    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = clf.predict(X_train)
        y_pred_val = clf.predict(X_validate)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_validate, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for Decision Tree classifier on training set:   {train_score:.4f}')
        print(f'{method} for Decision Tree classifier on validation set: {val_score:.4f}')
        print(classification_report(y_validate, y_pred_val))
    
    #return train_score, val_score
    
    

def knn_mod(X_train, y_train, X_validate, y_validate, metric = 1, print_scores = False):
    """
    This function runs the KNN classifier on the training and validation test sets.
    """
    #Creating the model
    knn = KNeighborsClassifier(n_neighbors=2, weights='uniform')

    #Fitting the KNN model
    knn.fit(X_train, y_train)

    #Accuracy
    if metric == 1:
        train_score = knn.score(X_train, y_train)
        val_score =  knn.score(X_validate, y_validate)
        y_pred_val = knn.predict(X_validate)

        method = 'Accuracy'

    #Precision
    elif metric == 2:
        #Make a prediction from the model
        y_pred = knn.predict(X_train)
        y_pred_val = knn.predict(X_validate)

        train_score = precision_score(y_train, y_pred, average='micro')
        val_score = precision_score(y_validate, y_pred_val, average='micro')
        method = 'Precision'

    #Recall
    elif metric == 3:
        
        #Make a prediction from the model
        y_pred = knn.predict(X_train)
        y_pred_val = knn.predict(X_validate)

        train_score = recall_score(y_train, y_pred, average='micro')
        val_score = recall_score(y_validate, y_pred_val, average='micro')
        method = 'Recall'
        
    #Print the score
    if print_scores == True:
        print(f'{method} for KNN classifier on training set:   {train_score:.4f}')
        print(f'{method} for KNN classifier on validation set: {val_score:.4f}')
        print(classification_report(y_validate, y_pred_val))

    #return train_score, val_score


#################################################### Top ML Models Accuracy Function ####################################################


def get_top_acc_models(X_train, y_train, X_validate, y_validate):
    '''This function gets all the top ML models for accuracy scores and plots them together for a visual'''

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
    plt.ylabel('Accuracy Scores')
    plt.title('Top Accuracy Models')
    sns.set_theme(style="whitegrid")
    ax.annotate('Best Model',fontsize=12,color="Black",weight="bold", xy=(1, 1), 
                xytext=(.65, .9))
    mylabels = ['Train', 'Validate']
    ax.legend(labels=mylabels,bbox_to_anchor=(1.02, 1), loc='upper left',borderaxespad=0)
    plt.show()


#################################################### Top ML Models Recall Function ####################################################


def get_top_recall_models(X_train, y_train, X_validate, y_validate):
    '''This function gets all the top ML models for recall scores and plots them together for a visual'''

    ##### Best Logistic Regression #####
    best_lr_recall = LogisticRegression(C=.1, random_state=42,intercept_scaling=1, 
                                        solver='newton-cg')
    #Fit the model to the train data
    best_lr_recall.fit(X_train, y_train)

    #Make a prediction from the model
    y_pred = best_lr_recall.predict(X_train)
    y_pred_val = best_lr_recall.predict(X_validate)
    
    # Make best variables to 
    best_lr_train_recall = recall_score(y_train, y_pred, average='micro')
    best_lr_val_recall = recall_score(y_validate, y_pred_val, average='micro')

    ##### Best Random Forest #####
    best_rf_recall = RandomForestClassifier(max_depth=5, random_state=42, class_weight='balanced', 
                                n_estimators=100, min_samples_leaf=5)
    #Fit the model to the train data
    best_rf_recall.fit(X_train, y_train)

    #Make a prediction from the model
    y_pred = best_rf_recall.predict(X_train)
    y_pred_val = best_rf_recall.predict(X_validate)

    best_rf_train_recall = recall_score(y_train, y_pred, average='micro')
    best_rf_val_recall = recall_score(y_validate, y_pred_val, average='micro')

    ##### Best KNN #####
    best_knn_recall = KNeighborsClassifier(n_neighbors=2, weights='uniform')
    #Fitting the KNN model
    best_knn_recall.fit(X_train, y_train)

    #Make a prediction from the model
    y_pred = best_knn_recall.predict(X_train)
    y_pred_val = best_knn_recall.predict(X_validate)

    best_knn_train_recall = recall_score(y_train, y_pred, average='micro')
    best_knn_val_recall = recall_score(y_validate, y_pred_val, average='micro')

    ##### Best Decision Tree #####
    best_clf_recall = DecisionTreeClassifier(max_depth=5, random_state=42)    
    #Train the model
    best_clf_recall.fit(X_train, y_train)
    
    #Make a prediction from the model
    y_pred = best_clf_recall.predict(X_train)
    y_pred_val = best_clf_recall.predict(X_validate)

    best_clf_train_recall = recall_score(y_train, y_pred, average='micro')
    best_clf_val_recall = recall_score(y_validate, y_pred_val, average='micro')

    # lists with model names & score information
    best_model_name_list = ["Logistic_Regression","Random_Forest","KNN","Decision Tree"]
    best_model_train_recall_list = [best_lr_train_recall,best_rf_train_recall,
                                    best_knn_train_recall,best_clf_train_recall]
    best_model_validate_recall_list = [best_lr_val_recall,best_rf_val_recall,
                                       best_knn_val_recall,best_clf_val_recall]
    
    # new empty DataFrame
    best_scores_df = pd.DataFrame()

    # new columns using lists for data
    best_scores_df["Model"] = best_model_name_list
    best_scores_df["Train_Recall"] = best_model_train_recall_list
    best_scores_df["Validate_Recall"] = best_model_validate_recall_list

    # plot it
    plt.figure(figsize=(11, 8.5))
    ax = best_scores_df.plot.bar(rot=5)
    plt.xticks(np.arange(4), ['Logistic Regression', 'Random Forest','KNN', 'Decision Tree'])
    plt.ylabel('Recall Score')
    plt.title('Top Recall Models')
    sns.set_theme(style="whitegrid")
    ax.annotate('Best Model',fontsize=12,color="Black",weight="bold", xy=(1, 1), 
                xytext=(2.65, .9))
    mylabels = ['Train', 'Validate']
    ax.legend(labels=mylabels,bbox_to_anchor=(1.02, 1), loc='upper left',borderaxespad=0)
    plt.show()


#################################################### Test Functions ####################################################
    

def get_acc_test(X_train, y_train, X_test, y_test):
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
    


def get_recall_test(X_train, y_train, X_test, y_test):
    '''
    This function gets our best peforming model and runs it on our test data
    '''
    # best decision tree
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    # Calculate recall score
    y_pred = clf.predict(X_test)
    recall = recall_score(y_test, y_pred, average='micro')
    # clean f string
    display(Markdown(f'### Decision Tree Model'))
    display(Markdown(f'### Recall Score On Test {(recall) * 100:.2f}%'))    



def get_mvb(X_train, y_train, X_test, y_test, df):
    '''This function plots the test data and plot the baseline together for a final visual'''
    
    # Recalculating Best Peforming Model with new name
    best_model = RandomForestClassifier(max_depth=5, random_state=42,
                            max_samples=0.5)
    best_model.fit(X_train, y_train)  
    best_model.score(X_test,y_test)
    
    # best decision tree
    clf= DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    # Calculate recall score
    y_pred = clf.predict(X_test)
    recall = recall_score(y_test, y_pred, average='micro')
    
    # Baseline
    plot_baseline = (df['fraud'] == df['baseline']).mean() 
    
    # Best Performing Model(Logistic Regression Combo{c=100,newton-cg}) Test Score: 
    best_test_score = best_model.score(X_test,y_test)  
    
    # Test Scores: Project Baseline vs Best Model
    plot_baseline, best_test_score, recall
    
    # Temporary Dictionary Holding Baseline & Model Test Score
    best_model_plot={"Baseline":[plot_baseline], "Test Accuracy":[best_test_score], "Test Recall":[recall]}
    
    # Converting Temporary Dictionary to DataFrame
    best_model_plot = pd.DataFrame(best_model_plot)
    
    # Visualizing Both Baseline & Model Test Scores
    fig=sns.barplot(data= best_model_plot,palette="colorblind")
    plt.title("Best Model vs. Baseline")
    fig.set(ylabel='Scores')
    plt.show()




