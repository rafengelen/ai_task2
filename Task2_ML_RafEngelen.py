
import pandas as pd

from sklearn.model_selection import train_test_split

import category_encoders as ce

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from io import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

import streamlit as st
def print_confusion(actual, prediction):

    confusion = confusion_matrix(actual, prediction, labels = ["positive", "negative"])
    print(f"Confusion matrix: \n{confusion}")
    st.write(f"Confusion matrix: \n{confusion}")

    # Good predictions: 
    correct_predictions = confusion.diagonal().sum()
    print(f"Amount of correct predictions: {correct_predictions}")
    st.write(f"Amount of correct predictions: {correct_predictions}")

    # Accuracy:
    print(f"Accuracy: {accuracy_score(actual, prediction)}")
    st.write(f"Accuracy: {accuracy_score(actual, prediction)}")
    # Ook mogelijk voor accuracy: print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

from sklearn.naive_bayes import GaussianNB, CategoricalNB

from sklearn.neural_network import MLPClassifier

if 'models_trained' not in st.session_state:
    st.session_state['models_trained'] = False
    
@st.cache(allow_output_mutation=True)
def main():
    feature_cols=[
    "top-left", "top-middle", "top-right", 
    "middle-left", "middle-middel", "middle-right", 
    "bottom-left", "bottom-middle", "bottom-right" 
    ]
    tictactoe_df = pd.read_csv("data/tic-tac-toe.data",
                           sep=',',
                           header=None, 
                           names=feature_cols+["x has won"]
                           )
    
    X = tictactoe_df[feature_cols] # Features
    y = tictactoe_df[['x has won']] # target variable
    
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

    ce_oh = ce.OneHotEncoder(cols = feature_cols)
    X_train_cat_oh = ce_oh.fit_transform(X_train)
    X_test_cat_oh = ce_oh.fit_transform(X_test)

    clf_baseline = DecisionTreeClassifier(criterion = "entropy").fit(X_train_cat_oh, y_train)
    y_pred_baseline = clf_baseline.predict(X_test_cat_oh)

    clf_gnb= GaussianNB().fit(X_train_cat_oh, y_train)
    y_pred_gnb = clf_gnb.predict(X_test_cat_oh)

    clf_mlp = MLPClassifier().fit(X_train_cat_oh, y_train)
    y_pred_mlp = clf_mlp.predict(X_test_cat_oh)

    return y_test, y_pred_baseline, y_pred_gnb, y_pred_mlp




# Train the models only if they haven't been trained yet
if not st.session_state.models_trained:
    y_test, y_pred_baseline, y_pred_gnb, y_pred_mlp = main()
    st.session_state.models_trained = True  # Set the flag to True after training the models


st.header('Raf Engelen - r0901812 - 3APP01', divider='gray')
st.title("Task 2 ML: Benchmarking two ML algorithms")
option = st.sidebar.selectbox(
    'Choose machine learning model',
    ('Decision Tree', 'Gaussian Naive Bayes', 'Multi-layer Perceptron')
)
st.subheader('Information about the trained model')
if option == 'Decision Tree':
    print_confusion(y_test, y_pred_baseline)
elif option == 'Gaussian Naive Bayes':
    print_confusion(y_test, y_pred_gnb)
elif option == 'Multi-layer Perceptron':
    print_confusion(y_test, y_pred_mlp)

# %% [markdown]
# ## Bronnenlijst:
# 
# 1.9. Naive Bayes. (z.d.). scikit-learn. https://scikit-learn.org/stable/modules/naive_bayes.html
# 
# Dash, S. (2023, 3 november). Decision Trees explained — entropy, information gain, Gini index, CCP pruning. Medium. https://towardsdatascience.com/decision-trees-explained-entropy-information-gain-gini-index-ccp-pruning-4d78070db36c 
# 
# Sethi, A. (2023, 15 juni). One Hot Encoding vs. label encoding using SciKit-Learn. Analytics Vidhya. https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/#:~:text=Label%20encoding%20is%20simpler%20and,lead%20to%20high-dimensional%20data 
# 
# UCI Machine Learning Repository. (z.d.). https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame
# 
# Zach. (2022, 8 augustus). Label encoding vs. one hot encoding: What’s the difference? Statology. https://www.statology.org/label-encoding-vs-one-hot-encoding/
# 
# Vats, R. (z.d.). Top 12 Commerce Project Topics & Ideas in 2023 [For Freshers]. upGrad blog. https://www.upgrad.com/blog/gaussian-naive-bayes/
# 
# Bento, C. (2022, 5 januari). Multilayer Perceptron explained with a Real-Life example and Python code: Sentiment analysis. Medium. https://towardsdatascience.com/multilayer-perceptron-explained-with-a-real-life-example-and-python-code-sentiment-analysis-cb408ee93141


