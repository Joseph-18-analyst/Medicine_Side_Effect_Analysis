"""
Created on Mon Mar  2 11:21:53 2020

@author: Medha
# EDA for medicine side effects
"""


 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import nltk 
import re
from textblob import TextBlob
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 


medicineDf=pd.read_csv("C:\\Users\\Joseph\\Documents\\Excel_R_Project\\train.csv")
 

"""
#################  EDA   ################
"""
medicineDf.columns
str(medicineDf)
medicineDf.info()
medicineDf.describe()
medicineDf.head(20)
medicineDf.isnull().sum()
medicineDf.isnull()

plt.hist(medicineDf.rating)


#Drug Count
drugs = pd.value_counts(medicineDf.drugName)
drugs.head(20)
drugs[drugs == drugs.min()].head(20)

fig = plt.figure(figsize = (80,25))
sns.countplot(drugs)

#Rating Count
ratingValue = pd.value_counts(medicineDf.rating)

rateDf = ratingValue.to_frame()
sns.catplot(x= rateDf.index,y= rateDf.rating,data=rateDf)

sns.catplot(x="drugName",y="rating",jitter = False,data = medicineDf)

rating_count=medicineDf.groupby('rating').count()
plt.bar(rating_count.index.values, rating_count['review'])
plt.xlabel('Review')
plt.ylabel('Number of Review')
plt.show()


#Condition Count
conditions = pd.value_counts(medicineDf.condition)
sns.countplot(conditions)

#Maximum number of useful count 
medicineDf.usefulCount.idxmax() 
use = max(medicineDf.usefulCount)
 
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
ps = PorterStemmer() 
 

drugsGrp = medicineDf.groupby(['drugName'])
f = drugsGrp.first()
#p = drugsGrp.get_group('Ibuprofen')
#review = p.review
drugGroup = drugsGrp.describe()
drugGroup.columns
#drugGroup.drop(drugGroup.columns[[0]],axis=1,inplace = True)

drugDTMdf = pd.DataFrame(columns = ['drugName' , 'Review'])
drugDTMdf = drugDTMdf[0:0]
"""
################ EDA DONE ##############
"""

review = ' '
stem_word = ' '
#### Filter the words for a particular drug name
#review = medicineDf.review[0:199]
for drugName, group in drugsGrp:
 
            p = drugsGrp.get_group(drugName)
            review = p.review

            filtered_list= []
            for r in review:
                filtered_word=[]
                w = 0
                tokenized_word=word_tokenize(r) 
                tokenized_word = [x.lower() for x in tokenized_word] 
                for w in tokenized_word:
                    if w not in stop_words:
                        stem_word = ps.stem(w)
                        filtered_word.append(stem_word)

                filtered_word = ["".join(list(filter(str.isalnum, line))) for line in filtered_word]
                pattern = '[0-9]'
                filtered_word = [re.sub(pattern, '', i) for i in filtered_word] 
                filtered_word = [i for i in filtered_word if i] 
                filtered_list.extend(filtered_word)
            
            comment_words = ' '
            review = ' '
            
            #Combine all reviews for a drug in one text
            for val in filtered_list:       
                # typecaste each val to string 
                val = str(val)   
                # split the value 
                tokens = val.split()       
                # Converts each token into lowercase 
                for i in range(len(tokens)): 
                    tokens[i] = tokens[i].lower()           
                for words in tokens: 
                    comment_words = comment_words + words + ' '
#            print(drugName)
#            print(comment_words)
            drugDTMdf = drugDTMdf.append({'drugName' : drugName , 'Review' : comment_words} , ignore_index=True)

#Get ratingsfrom grouped rating table. consiered 75% rating 
rating = drugGroup.get('rating',   '75%')
drugDTMdf['Rating'] = rating['75%'].values

# Defining a sentiment analyser function
def sentiment_analyser(text):
    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))

# Applying function to reviews
drugDTMdf['Polarity'] = sentiment_analyser(drugDTMdf['Review'])
drugDTMdf.head(10)

#Lemmatization#################################################3333
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

# Define a word lemmatizer function
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

# Apply the word lemmatizer function to data
drugDTMdf['Filtered_Review_Text'] = drugDTMdf['Review'].apply(lemmatize_text)
drugDTMdf.head()
#drugDTMdf.columns
#drugDTMdf.drop(drugDTMdf.columns[[4]],axis=1,inplace = True)

Filtered_Review_String = []
comment_words = ' '
words = ' '
 
for val in drugDTMdf.Filtered_Review_Text: 
    
        for words in val: 
            comment_words = comment_words + words + ' '
            
        Filtered_Review_String.append(comment_words)
        comment_words = ' '
        words = ' '
    
        
drugDTMdf['Filtered_Review_String'] = Filtered_Review_String        
drugDTMdf.columns        
# Applying function to reviews
drugDTMdf['Polarity'] = sentiment_analyser(drugDTMdf['Filtered_Review_String'])

# Getting a count of words from the documents
# Ngram_range is set to 1,2 - meaning either single or two word combination will be extracted
cvec = CountVectorizer(min_df=.005, max_df=.9, ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False)
cvec.fit(drugDTMdf['Filtered_Review_Text'])

# Getting the total n-gram count
len(cvec.vocabulary_)

# Creating the bag-of-words representation
cvec_counts = cvec.transform(drugDTMdf['Filtered_Review_Text'])
print('sparse matrix shape:', cvec_counts.shape)
print('nonzero count:', cvec_counts.nnz)
print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))

# Instantiating the TfidfTransformer
transformer = TfidfTransformer()

# Fitting and transforming n-grams
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights

# Getting a list of all n-grams
transformed_weights = transformed_weights.toarray()
vocab = cvec.get_feature_names()

# Putting weighted n-grams into a DataFrame and computing some summary statistics
model = pd.DataFrame(transformed_weights, columns=vocab)
model['Keyword'] = model.idxmax(axis=1)
model['Max'] = model.max(axis=1)
model['Sum'] = model.drop('Max', axis=1).sum(axis=1)
model.head(10)

# Merging td-idf weight matrix with original DataFrame
model = pd.merge(drugDTMdf, model, left_index=True, right_index=True)

# Printing the first 10 reviews left
model.head(5)

# Getting a view of the top 20 occurring words
occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Occurrences': occ})
counts_df.sort_values(by='Occurrences', ascending=False).head(25)

# Getting a view of the top 20 weights
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'Term': cvec.get_feature_names(), 'Weight': weights})
weights_df.sort_values(by='Weight', ascending=False).head(25)

#Round off Rating 
model['Rating'] = [round(x, 0) for x in model['Rating']]
model["Category"] = 0

model.loc[model['Rating'] > 6, 'Category'] = 1 
model.loc[model['Rating'] <= 6, 'Category'] = 0


# Plotting overall recommendations and getting value counts
fig = plt.figure(figsize = (10,5))
sns.countplot(x='Category', data = model)
plt.xticks(rotation=90)
print(drugDTMdf['Category'].value_counts())
 

# Visualising polarity between recommending and non-recommending customers, then getting value counts
g = sns.FacetGrid(model, col="Category", col_order=[1,0])
g = g.map(plt.hist, "Polarity", bins=20, color="g")

recommend = model.groupby(['Category'])
recommend['Polarity'].mean()
  
# Get a list of columns for deletion
model.columns

modelSave = model
# Drop all columns not part of the text matrix
ml_model = model.drop(['drugName', 'Review', 'Rating', 'Filtered_Review_Text','Filtered_Review_String', 'Polarity', 'Keyword', 'Max', 'Sum'], axis=1)

# Create X & y variables for Machine Learning
X = ml_model.drop('Category', axis=1)
y = ml_model['Category']

# Create a train-test split of these variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)


# Defining a function to fit and predict ML algorithms
def model(mod, model_name, x_train, y_train, x_test, y_test):
    mod.fit(x_train, y_train)
    print(model_name)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, X_train, y_train, cv = 5)
    print("Accuracy:", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train)
    print("Confusion Matrix:  \n", cm)
    print("                    Classification Report \n",classification_report(predictions, y_train))
    
    
 # 1. Gaussian Naive Bayes
gnb = GaussianNB()
model(gnb, "Gaussian Naive Bayes", X_train, y_train, X_test, y_test)   

# 2. Random Forest Classifier
ran = RandomForestClassifier(n_estimators=50)
model(ran, "Random Forest Classifier", X_train, y_train, X_test, y_test)

# 3. Logistic Regression
log = LogisticRegression()
model(log, "Logistic Regression", X_train, y_train, X_test, y_test)

# 4. Linear SVC
svc = LinearSVC()
model(svc, "Linear SVC", X_train, y_train, X_test, y_test)

# Import the hopeful solution to our problems
from imblearn.over_sampling import SMOTE
smote=SMOTE()

# Setting up new variables for ML
X_sm, y_sm = smote.fit_sample(X,y)

X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.3, random_state=100)

# Defining a new function with revised inputs for the new SMOTE variables
def model_sm(mod, model_name, x_train_sm, y_train_sm, x_test_sm, y_test_sm):
    mod.fit(x_train_sm, y_train_sm)
    print(model_name)
    acc = cross_val_score(mod, X_train_sm, y_train_sm, scoring = "accuracy", cv = 5)
    predictions = cross_val_predict(mod, X_train_sm, y_train_sm, cv = 5)
    print("Accuracy:", round(acc.mean(),3))
    cm = confusion_matrix(predictions, y_train_sm)
    print("Confusion Matrix:  \n", cm)
    print("                    Classification Report \n",classification_report(predictions, y_train_sm))
    
# 1. Gaussian Naive Bayes
gnb = GaussianNB()
model_sm(gnb, "Gaussian Naive Bayes", X_train_sm, y_train_sm, X_test_sm, y_test_sm)    
    
# 2. Random Forest Classifier
ran = RandomForestClassifier(n_estimators=50)
model_sm(ran, "Random Forest Classifier", X_train_sm, y_train_sm, X_test_sm, y_test_sm)    

# 3. Logistic Regression
log = LogisticRegression()
model_sm(log, "Logistic Regression", X_train_sm, y_train_sm, X_test_sm, y_test_sm)

# 4. Linear SVC
svc = LinearSVC()
model_sm(svc, "Linear SVC", X_train_sm, y_train_sm, X_test_sm, y_test_sm)

# Creating a plot for feature importance
def importance_plotting(data,x,y,palette,title):
    sns.set(style="whitegrid")
    ft = sns.PairGrid(data,y_vars=y,x_vars=x,size=5,aspect=1)
    ft.map(sns.stripplot,orient='h',palette=palette, edgecolor="black",size=15)
    for ax, title in zip(ft.axes.flat, titles):
        
    # Set a different title for each axes
        ax.set(title=title)
        
    # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

    plt.show()
    
# Compile arrays of columns (words) and feature importances
fi = {'Words':ml_model.drop('Category',axis=1).columns.tolist(),'Importance':ran.feature_importances_}

# Bung these into a dataframe, rank highest to lowest then slice top 20
Importance = pd.DataFrame(fi,index=None).sort_values('Importance',ascending=False).head(25)

# Plot the graph!
titles = ["Top 25 most important words in predicting product recommendation"]
importance_plotting(Importance,'Importance','Words','Greens_r',titles)
 
"""   
###############################################################
#Logistic Regression Algo gives more accurancy    
# Getting prediction probabilities
y_scores = log.predict_proba(X_train_sm)
###############################################################
""" 
y_scores = log.predict_proba(X_train_sm)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(y_train_sm, y_scores)    


# Defining a new function to plot the precision-recall curve
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("Threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.show()

# Compute the true positive and false positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_sm, y_scores)


# Plotting the true positive and false positive rate
def plot_roc_curve(false_positive_rate, true_positive_rate, label=None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'r', linewidth=4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize=16)
    plt.ylabel('True Positive Rate (TPR)', fontsize=16)

plt.figure(figsize=(14, 7))
plot_roc_curve(false_positive_rate, true_positive_rate)
plt.show()

#another way to plot ROC curve
# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_train_sm))]
# calculate scores
ns_auc = roc_auc_score(y_train_sm, ns_probs)
lr_auc = roc_auc_score(y_train_sm, y_scores)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_train_sm, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_train_sm, y_scores)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()



# Computing the ROC-AUC score
r_a_score = roc_auc_score(y_train_sm, y_scores)
print("ROC-AUC-Score:", r_a_score)


