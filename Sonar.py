import pandas as pd
from sklearn.model_selection import train_test_split
#file import
path = ""
df = pd.read_csv(path + "sonar_hw1.csv")
df.head()
y = []
#dropping the target value
y = df['object']
df = df.drop('object', axis=1)


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


X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=4)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)
###############################################################################
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

X, y = load_iris(return_X_y = True)
clf = LogisticRegression(random_state = 123, solver = 'lbfgs', 
                         multi_class ='multinomial', max_iter = 1000).fit(X,y)

clf.predict(df[:2,:])
clf.predict_proba(df[:2, :])
clf.score(X,y)
