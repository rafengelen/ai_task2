
import pandas as pd
import category_encoders as ce
from sklearn.metrics import confusion_matrix,accuracy_score
from io import StringIO
from IPython.display import Image
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import streamlit as st
def print_confusion(actual, prediction):

    confusion = confusion_matrix(actual, prediction, labels = ["positive", "negative"])
    print(f"Confusion matrix: \n{confusion}")
    st.write(f"Confusion matrix: \n{confusion}")

    # Good predictions: 
    correct_predictions = confusion.diagonal().sum()
    print(f"Amount of correct predictions: {correct_predictions}")
    st.write(f"Amount of correct predictions: \n{correct_predictions}")

    # Accuracy:
    print(f"Accuracy: {accuracy_score(actual, prediction)}")
    st.write(f"Accuracy: \n{accuracy_score(actual, prediction)}")
    # Ook mogelijk voor accuracy: print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred)}")

from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neural_network import MLPClassifier

@st.cache(allow_output_mutation=True)
def train_models():
    # Load your data (replace this with your data loading code)
    data = pd.read_csv("data/tic-tac-toe.data",
                       sep=',',
                       header=None,
                       names=[
                           "top-left", "top-middle", "top-right",
                           "middle-left", "middle-middel", "middle-right",
                           "bottom-left", "bottom-middle", "bottom-right",
                           "x has won"]
                       )

    feature_cols = [
        "top-left", "top-middle", "top-right",
        "middle-left", "middle-middel", "middle-right",
        "bottom-left", "bottom-middle", "bottom-right"
    ]

    X = data[feature_cols]  # Features
    y = data[['x has won']]  # target variable

    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test

    # One Hot Encoding
    ce_oh = ce.OneHotEncoder(cols=feature_cols)
    X_train_cat_oh = ce_oh.fit_transform(X_train)
    X_test_cat_oh = ce_oh.fit_transform(X_test)

    # Train models
    clf_baseline = DecisionTreeClassifier(criterion="entropy")
    clf_baseline = clf_baseline.fit(X_train_cat_oh, y_train)

    clf_gnb = GaussianNB().fit(X_train_cat_oh, y_train)

    clf_mlp = MLPClassifier().fit(X_train_cat_oh, y_train)

    return clf_baseline, clf_gnb, clf_mlp, X_test_cat_oh, y_test

clf_baseline, clf_gnb, clf_mlp, X_test_cat_oh, y_test = train_models()

def make_predictions(clf_baseline, clf_gnb, clf_mlp, X_test_cat_oh):
    return clf_baseline.predict(X_test_cat_oh), clf_gnb.predict(X_test_cat_oh), clf_mlp.predict(X_test_cat_oh)
y_pred_baseline, y_pred_gnb, y_pred_mlp = make_predictions(clf_baseline, clf_gnb, clf_mlp, X_test_cat_oh)
    
st.header('Raf Engelen - r0901812 - 3APP01', divider='gray')
st.title("Task 2 ML: Benchmarking two ML algorithms")

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Train the models only if they haven't been trained yet
if not st.session_state.models_trained:
    st.session_state.models_trained = True  # Set the flag to True after training the models

option = st.sidebar.selectbox(
    'Choose machine learning model',
    ('Decision Tree', 'Gaussian Naive Bayes', 'Multi-layer Perceptron')
)
toggle_button = st.sidebar.button('Toggle Data Visibility')





if option == 'Decision Tree':
    st.subheader('Decision Tree Model Information')
    df_data = pd.concat([y_test, pd.DataFrame({'prediction': y_pred_baseline}, index=y_test.index)], axis=1).rename(columns={"x has won": "Actual"}, inplace=True) 
    print_confusion(y_test, y_pred_baseline)

elif option == 'Gaussian Naive Bayes':
    st.subheader('Gaussian Naive Bayes Model Information')
    y_pred_gnb = clf_gnb.predict(X_test_cat_oh)
    print_confusion(y_test, y_pred_gnb)

elif option == 'Multi-layer Perceptron':
    st.subheader('Multi-layer Perceptron Model Information')
    y_pred_mlp = clf_mlp.predict(X_test_cat_oh)
    print_confusion(y_test, y_pred_mlp)

if toggle_button:
    # Toggle the visibility state
    st.session_state.data_visible = not st.session_state.data_visible if 'data_visible' in st.session_state else True

    if st.session_state.data_visible:
        # Display the DataFrame (replace this with your DataFrame display code)
        st.write(st.write(df_data))
else:
    st.session_state.data_visible = False  # Set data_visible to False if button is not pressed
    




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


