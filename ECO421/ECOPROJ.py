import pandas as pd
import numpy as np
import decimal
import matplotlib.pyplot as plt
import string
import nltk
import statistics
import statsmodels.api as sm

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from scipy.stats.stats import pearsonr   
from sklearn.linear_model import LinearRegression
from scipy.stats import chisquare
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import cross_val_score

#------------------

#Q1: Read file and clean data

df = pd.read_csv(r'C:\Users\tiffa\Downloads\ECOPROJECT\vax_data.csv', encoding = 'cp1252')
#Convert df to string since drop_dup doesn't work for objects

#print(df['education'].unique())
#print(df['trusthealth'])
#print(df['education'].value_counts())

#Convert strings to numeric
df['education'] = pd.to_numeric(df.education, errors='coerce', downcast = 'integer')
df['age'] = pd.to_numeric(df.age, errors='coerce',downcast = 'integer')
df['gender'] = pd.to_numeric(df.gender, errors='coerce', downcast = 'integer')

#Replace all non-responses to 0
num_response = [1,2,3,4,5]
df.loc[~df["education"].isin(num_response), "education"] = 0
df.loc[~df["gender"].isin(num_response), "gender"] = 0
df.loc[~df["age"].isin(num_response), "age"] = 0
df.loc[~df["vaccine"].isin(num_response), "vaccine"] = 0



#Variable Names
edu = df['education']
age = df['age']
gender = df['gender']
vaccine = df['vaccine']

#Data Summary
#---------------
df.info()

#Histogram Plots
#---
plt.hist(edu)
plt.title('Education Level Distribution', fontsize=14)
plt.xlabel('Education', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(range(0,5)) #makes xaxis integers. Figure out how to make bars not weird lol
plt.show()

plt.hist(age)
plt.title('Age Group Distribution', fontsize=14)
plt.xlabel('Age', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

plt.hist(gender)
plt.title('Gender Distribution', fontsize=14)
plt.xlabel('Education', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

plt.hist(vaccine)
plt.title('Vaccine Distribution', fontsize=14)
plt.xlabel('Education', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

#Scatterplot
#---------------
#Education vs Vaccine
N = 717
df_new  = df.loc[0:718,['education','age','gender','vaccine']]
s = (25 * np.random.rand(N))**2
plt.scatter(edu, vaccine, s = s, c = 'blue', alpha = 0.05)
plt.show()
plt.xlabel("Education Level", size=12)
plt.ylabel("Vaccine", size=12)
plt.title("Education vs Vaccine Acceptance", size=5)
#Most common correlation is Edu (3,4) to Vaccine (5), and Edu(1) to Vaccine (3)

#Age vs Vaccine
N = 717
s = (25 * np.random.rand(N))**2
plt.scatter(age, vaccine, s = s, c = 'blue', alpha = 0.03)
plt.show()
plt.xlabel("Age Group", size=12)
plt.ylabel("Vaccine", size=12)
plt.title("Age vs Vaccine Acceptance", size=5)
#Most common correlation is Age (1,2) to Vaccine (5) and older ages less willing

#Gender vs Vaccine
N = 717
s = (25 * np.random.rand(N))**2
plt.scatter(gender, vaccine, s = s, c = 'blue', alpha = 0.03)
plt.show()
plt.xlabel("Gender", size=12)
plt.ylabel("Vaccine", size=12)
plt.title("Gender vs Vaccine Acceptance", size=12)
#Most common correlation is Gender (1) to Vaccine (5) Gender (2) to Vaccine (5)


#Correlation Matrix
#----------------------
corr = df_new.corr()
corr
#Highest correlation is between education and vaccine (0.3), lowest is gender and vaccine (0.02)

#Pearson Chi-Squared Test
#------------
#Use Pearon's Chi-Squared Test for categorical data to evaluated observed differences between sets 
#Tells us whether two variables are ID of one another
#Null Hypothesis: frequency dist of certain events observed in sample consistent with theoretical 

#df=df_new.apply(lambda x : pd.factorize(x)[0])+1

#pd.DataFrame([chisquare(df[x].values,f_exp=df.values.T,axis=1)[0] for x in df])
#0.19 (edu), 0.34 (age), 0.26 (gender)

#Logistic Regression
#---------------
#Split data into train and test
features = ['education','gender','age']
x = df_new[features]
y = df_new['vaccine']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1) #default split in 70/30
#x_train.shape

#Scale data 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
#print(log_reg.coef_)
#print(log_reg.intercept_)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(log_reg.score(x_train, y_train))) #0.61
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(log_reg.score(x_test, y_test))) #0.59


#Decision Tree
#---------------
clf = DecisionTreeClassifier().fit(x_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(x_train, y_train))) #0.65
print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(x_test, y_test))) #0.64

#KNN Neighbours
#--------------
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(x_train, y_train))) #0.59
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(x_test, y_test))) #0.56

#Naive Bayes
#---------------
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(x_train, y_train))) #0.63
print('Accuracy of GNB classifier on test set: {:.2f}'
     .format(gnb.score(x_test, y_test))) #0.62

#Decision Tree - (PRELIMINARY)
#-----------------
df_new  = df.loc[1:717,['education','age','gender','vaccine']]
features = ['education','gender','age']
x = df_new[features]
y = df_new['vaccine']
#convert floats to categorical (one hot encoding)
x_enc = pd.get_dummies(x, columns = features)
x_enc.head()

#Fit DT to training data
x_train, x_test, y_train, y_test = train_test_split(x_enc, y)

clf_dt = DecisionTreeClassifier()
clf_dt = clf_dt.fit(x_train, y_train)
classes = ['Disagree','Neutral','Agree']
plt.figure(figsize = (15,7.5))
plot_tree(clf_dt, filled = True, rounded = True, class_names = classes, feature_names = x_enc.columns)

#Confusion Matrix (UNPRUNED)
plot_confusion_matrix(clf_dt, x_test, y_test, display_labels = classes)

#Classification Trees may be overfitting the training dataset. So we should prune to solve overfitting problem and give better results.
#Decision trees known to overfit so we should use param max, min to reduce this. We should find smaller tree to improve accuracy of test 
#Must find optimal pruning parameter alpha (controls how little/much purning happens). 
#We find optimal alpha by plotting accuracy of tree as function of diff values (for both train/test dataset) 
#Extract diff alpha values, build pruned tree, omit max value of alpha because it prunes all leaves and gives root instead of tree 

#Decision Tree - (PRUNING)
#----------------

#Use CV to find optimal alpha
clf_dt = DecisionTreeClassifier(random_state = 1, ccp_alpha = 0.003)
#start with 10-KV because its standard
scores = cross_val_score(clf_dt, x_train, y_train, cv = 10)
df = pd.DataFrame(data = {'tree': range(10), 'accuracy': scores})
df.plot(x = "tree", y = "accuracy", marker = 'o')

#Graph shows diff train/test sets with same alpha have diff accuracies so alpha depends on dataset. We should use CV to find optimal ccp_alpha value so tree is not overfitting 

alpha_vals = []
path = clf_dt.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas #extract diff alpha values
ccp_alphas = ccp_alphas[:-1] #take out max alpha 

#create tree for each alpha, store in list
#(PLOT)
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(ccp_alpha = ccp_alpha)
    scores = cross_val_score(clf_dt, x_train, y_train, cv = 10)
    alpha_vals.append([ccp_alpha, np.mean(scores), np.std(scores)])
#optimal alpha seen at 0.003
alpha_results = pd.DataFrame(alpha_vals, columns = ['a','mean','sd'])
alpha_results.plot(x= 'a', y = 'mean', yerr = 'sd', marker = 'o')

#Find exact alpha value between range
opt_alpha = alpha_results[(alpha_results['a'] > 0.002) & (alpha_results['a'] < 0.005)]
final_alpha = max(opt_alpha['mean']) #0.6052410901467506

alpha_val = 0.003867 #optimal alpha is 0.003867

#FINAL TREE
#----------
pruned_tree = DecisionTreeClassifier(ccp_alpha = alpha_val)
pruned_tree = pruned_tree.fit(x_train, y_train)

#Confusion matrix (PRUNED)
plot_confusion_matrix(pruned_tree, x_test, y_test, display_labels = classes) #performs a little better at classifying

#Decision Tree Plot
plt.figure(figsize = (12,12))
plot_tree(pruned_tree, filled = True, rounded = True, class_names = classes, feature_names = x_enc.columns)

#Text Representation
text_rep = tree.export_text(pruned_tree)
print(text_rep)

