
import streamlit as st

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import silhouette_score

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import seaborn as sns

st.set_page_config(
    page_title="Kidney Disease Pridictor",
    page_icon="🏥",
)


df=pd.read_csv("data.csv")

st.title("KidneyInsight: Predictive Analytics for Chronic Kidney Disease ")
classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('LogisticRegression', 'KNN','Decision Tree' ,'Random Forest')
)
specific_gravity = st.number_input("Enter specific_gravity value", step=0.01, format="%.2f")
hemoglobin=st.number_input("Enter hemoglobin value",step=0.01, format="%.2f")
packed_cell_volume=st.number_input("Enter packed_cell_volume value", step=0.01, format="%.2f")
x = df[["specific gravity","hemoglobin","packed cell volume"]]
y = df["class"]

def get_classifier(clf_name):
    clf = None
    if clf_name == 'LogisticRegression':
        clf = LogisticRegression()
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=11)
    elif clf_name == "Decision Tree":
        clf= DecisionTreeClassifier()
    else:
        clf = RandomForestClassifier(n_estimators=10)
    return clf

clf = get_classifier(classifier_name)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



button = st.button("Predict", use_container_width=True)

# In[24]:


index = [0, 1, 2]
df_predict = pd.DataFrame({
    'specific gravity': specific_gravity,
    'hemoglobin': hemoglobin,
    'packed cell volume': packed_cell_volume}, index=index)

# In[25]:


if button:
    result = clf.predict(df_predict)

    if result[0] == 0:
        st.write("Patient has Chronic Kidney Disease (CKD)")
    elif result[0] == 1:
        st.write("Patient has Chronic Kidney Disease (CKD)")
    else:
        st.write("Patient not have Chronic Kidney Disease (CKD)")
    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    st.write("Classification Report")

    from sklearn.metrics import classification_report
    report_string = classification_report(y_test, y_pred)
    report_lines = report_string.split('\n')
    data = []
    for line in report_lines[2:-3]:  # Skip the first and last lines
        row_data = line.split()
        if len(row_data) >= 5:
            row_name = row_data[0]
            row_values = [float(x) for x in row_data[1:]]
            data.append([row_name] + row_values)

    # Create a DataFrame
    columns = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    df_report = pd.DataFrame(data, columns=columns)

    # Set the 'Class' column as the index
    df_report.set_index('Class', inplace=True)

    # Optionally, you can round the values for better presentation
    df_report = df_report.round(2)
    df_report



if classifier_name=="LogisticRegression":


    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    st.write('Confusion Matrix Heatmap:')
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot()

elif classifier_name == "Random Forest":
    # Plot the aggregated decision tree from Random Forest
    if hasattr(clf, "estimators_") and len(clf.estimators_) > 0:
        # Get the textual representation of each decision tree
        tree_rules = [tree.export_text(estimator, feature_names=["specific gravity", "hemoglobin", "packed cell volume"]) for estimator in clf.estimators_]

        # Combine the rules to form a summary tree
        summary_tree = "\n".join(tree_rules)

        # Plot the summary tree (as text)
        st.write("Aggregated Decision Tree from Random Forest:")
        st.code(summary_tree)
    else:
        st.write("Random Forest is empty or does not contain estimators.")
else:
    # Plot the decision boundary for KNN
    if len(x.columns) == 2:  # We can only visualize the decision boundary in 2D
        h = 0.01  # Step size in the mesh

        # Create a meshgrid to plot the decision boundary
        x_min, x_max = x.iloc[:, 0].min() - 1, x.iloc[:, 0].max() + 1
        y_min, y_max = x.iloc[:, 1].min() - 1, x.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the class for each point in the meshgrid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, cmap='rainbow', edgecolors='k')
        plt.xlabel('specific gravity')
        plt.ylabel('hemoglobin')
        plt.title('KNN Decision Boundary')
        plt.show()
        st.pyplot()
    else:
        st.write("Cannot visualize decision boundary for more than two features.")

















