import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

pima = pd.read_csv("diabetes.csv", )
col_names = list(pima.columns)
feature_cols = col_names[:len(col_names)-1]
  
X = pima[feature_cols]
y = pima['Outcome']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Build Decision Tree Model
clf = DecisionTreeClassifier()

# Train Decision Tree Classifier
clf = clf.fit(X_train, y_train)

# Make Predicitons 
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

print(feature_cols)
print(len(y))


from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    clf, 
    feature_names=feature_cols,
    filled=True,
    rounded=True,
    special_characters=True,
)

graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view()



