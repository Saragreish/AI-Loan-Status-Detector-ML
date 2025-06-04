# %%
import pandas as pd

# %%
dataset = pd.read_csv('D:\Downloads\loan_data.csv')

# %%
dataset.columns=dataset.columns.str.strip()

# %%
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 13].values

# %%
from sklearn.preprocessing import LabelEncoder
LabelEncoder_Y = LabelEncoder()
Y = LabelEncoder_Y.fit_transform(Y)

# %%
LabelEncoder_X = LabelEncoder()
X[:,1] = LabelEncoder_X.fit_transform(X[:, 2])
X[:,2]= LabelEncoder_X.fit_transform(X[:, 3])
X[:,5]= LabelEncoder_X.fit_transform(X[:, 6])
X[:,7]= LabelEncoder_X.fit_transform(X[:, 8])
X[:,-1]= LabelEncoder_X.fit_transform(X[:, -1])
print(X)

# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

# %%
from sklearn import tree
DT = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
y_pred = DT.fit(x_train, y_train).predict(x_test)

# %%
from sklearn.metrics import accuracy_score
tree_score = accuracy_score(y_pred, y_test)

# %% [markdown]
# 

# %%
import matplotlib.pyplot as plt
tree.plot_tree(DT)
plt.show()

# %%
print(f"Output of Decision Tree {tree_score}")


