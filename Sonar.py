import pandas as pd
from sklearn.model_selection import train_test_split

#file import
path = ""
df = pd.read_csv(path + "sonar_hw1.csv")
df.head()

##### Outlier Analysis #####

#Number of data points
n = df.shape[0] 
#missing cases
missing = n - pd.DataFrame(df.count(), columns = ['Missing'])
#outliers below 0
df_low = pd.DataFrame(df[df<0].count(), columns = ['Low outliers'])
#outliers above 1
df_high = pd.DataFrame(df[df>1].count(), columns = ['High outliers'])

#Boolean dataframe where True indicates not an outlier & not missing
df_accept = df[(df>=0) & (df<=1)]
#Minimum for only valid cases
minimum = pd.DataFrame(df_accept.min(), columns = ['Min'])
#Maximum for only valid cases
maximum = pd.DataFrame(df_accept.max(), columns = ['Max'])
#Median for only valid cases
median = pd.DataFrame(df_accept.median(), columns = ['Median'])

# Create list of dataframe names that will be joined into a single dataframe
df_list = [df_low, df_high, minimum, maximum, median]

#Initialize dataframe 
dataframe = missing
# loop over the list of dataframes joining them to create a single dataframe
for i in df_list:
    dataframe = dataframe.join(i)
print(dataframe)

##### Model fitting #####

X = df.drop(['R41','R46'], axis =1) #Dropping the whole Columns
D = X.dropna() #Dropping the missing value rows 
y = D['object']
X = D.drop('object', axis=1) #dropping the target value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=4)

print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)

# Logistic Regression #
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.
      format(logreg.score(X_test, y_test)))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#Precision, Recall, F score
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#ROC Curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# SVM #
from sklearn import svm

clf= svm.SVC(kernel='linear',C = 1.0,probability=True, random_state =12345)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('Accuracy of SVM classifier on test set: {:.2f}'.
      format(clf.score(X_test, y_test)))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#Precision, Recall, F score
print(classification_report(y_test, y_pred))
#ROC Curve
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='SVM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for SVM')
plt.legend(loc="lower right")
plt.savefig('Log1_ROC')
plt.show()
