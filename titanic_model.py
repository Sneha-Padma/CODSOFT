import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(f'C:/Users/sneha/OneDrive/Desktop/titanic_survival/Titanic-Dataset.csv')

df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
#Cleaning data done
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] =df['Embarked'].fillna(df['Embarked'].mode()[0])

le = LabelEncoder()
#encoding categorical var
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

#Splitting
x = df.drop('Survived', axis=1)
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42 )

#Training,Evaluating
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Logistic Regression Accuracy:",accuracy_score(y_test, y_pred))

#Visualization
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival Count by Gender")
plt.show()
