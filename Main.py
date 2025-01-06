importnumpy as np
import pandas as pd
importmatplotlib.pyplot as plt
importseaborn as sns

#Reading the data
df = pd.read_csv('creditcard.csv')
df.info()
df.drop(['Time','Amount'],axis=1,inplace=True)
print(df)

#Testing null values or missing data inform of graph
sns.heatmap(df.isnull())
plt.show()

#Testing for missing values
print(df.describe().T)
print(df.isnull().sum())

#Exploratory Data Analysis/Data visualization
sns.displot(df['V15'])
plt.show()

sns.boxenplot(data=df, x='Class', y='Amount')
plt.show()

sns.countplot('V5', data=df)
plt.show()

plt.hist(df)

plt.xlabel('Number of V');plt.ylabel('frequency'); plt.title('HISTOGRAM');
plt.show()

plt.figure(figsize=(17,17))

sns.heatmap(df.corr(), cmap='viridis', annot=True)
plt.show()

sns.scatterplot(data= df, x= 'V16', y='Amount',hue='Amount', alpha=0.7)
plt.show()

plt.figure(figsize=(20,8))
plt.scatter(x=df[df['Class'] == 1]['Time'], y=df[df['Class'] == 1]['Amount'], color="c", s=80)
plt.show()

figsize=(15, 5)
sns.boxplot(x='Class', y='V14', data=df)
plt.show()

#training the model
X = df.drop(['Time','Amount'], axis=1,inplace=True)
y = df['Class']
print(df.head())


#TESTING THE MODEL
fromsklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Class',axis=1), df['Class'],
test_size=0.25, random_state=42)

#LOGISTIC CLASSIFICATION MODEL
fromsklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

fromsklearn.metrics import classification_report, confusion_matrix, accuracy_score
accuracy_score(y_test, predictions)
print('Logistic algorithm')
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))

fromsklearn import metrics
print(metrics.r2_score(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))

#DECISION TREE ALGORITHM
fromsklearn.tree import DecisionTreeClassifier
d_tree = DecisionTreeClassifier()
d_tree.fit(X_train,y_train)
pred_dtree = d_tree.predict(X_test)

fromsklearn.metrics import classification_report, confusion_matrix, accuracy_score
print('Decision tree')
print(classification_report(y_test, pred_dtree))
print(confusion_matrix(y_test, pred_dtree))
print(accuracy_score(y_test,predictions))

# #RANDOM FOREST ALGORITHM
fromsklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
pred_rf = rfc.predict(X_test)
print('Random forest')
print(classification_report(y_test, pred_rf))
print(accuracy_score(y_test, pred_rf))
print(accuracy_score(y_test, pred_dtree))

#LINEAR REGRESSION ALGORITHM
fromsklearn.linear_model import LinearRegression
fromsklearn.metrics import r2_score
lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict( X_test)
r2_score(y_test, predictions)
print('Linear regression')
print(r2_score)

# #BAYESIAN MODEL
fromsklearn import linear_model
reg = linear_model.BayesianRidge()
reg.fit(X_train, y_train)
predic = reg.predict(X_test)
accuracy = r2_score(y_test,predic)
print('Bayesian')
print(np.round(accuracy,decimals=4))

# #STOSTAIC ALGORITHM
fromsklearn.linear_model import SGDRegressor
fromsklearn.pipeline import make_pipeline
fromsklearn.preprocessing import StandardScaler
n_samples, n_features = 20, 8
reg1 = make_pipeline(StandardScaler(),
SGDRegressor(max_iter=5000, tol=1e-3))
reg1.fit(X_train, y_train)
pr = reg1.predict(X_test)
accuracy = r2_score(y_test,pr)
print('stostatic')
print(np.round(accuracy,decimals=4))

#XGBoost algorithm
fromxgboost import XGBClassifier
fromsklearn.metrics import accuracy_score
xgb = XGBClassifier(max_depth = 4)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print('xgboost')
print('Accuracy score of the XGBoost model is {}'.format(accuracy_score(y_test,
xgb_pred)))
fromsklearn import metrics
LABELS = ['Normal', 'Fraud']
conf_matrix = metrics.confusion_matrix(y_test, xgb_pred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True,
fmt ="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
models = ['XGBoost','LogisticRegression','DecisionTree','Random Forest']
accuracy = [99.51405,99.46911,99.46911,98.97053]
r2score = ['Linear regression','Bayesian','stostatic']
r2 = [0.20582,0.1312,0.1253]

from matplotlib import style
style.use('ggplot')
plt.subplot(311)
plt.title('Accuracies')
plt.plot(models,accuracy)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.subplot(313)
plt.title('r2 Scores')
plt.scatter(r2score,r2,c='blue')
plt.xlabel('r2score')
plt.ylabel('r2')
plt.show()