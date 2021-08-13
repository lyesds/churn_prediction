import streamlit as st
# import churn_prediction.utils.c_supervised_classif
from utils.b_descriptive import piecharts, histos, scatters
from utils.c_supervised_classif import classify, probacluster
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


plt.style.use('seaborn-muted')

st.image('./assets/logo.png')
st.title("Bank customers churn prediction and clustering")
st.sidebar.title("Bank customers classification")
st.markdown("Welcome to this dashboard for bank customers insights! "
            "_Data courtesy of [sakshigoyal7@kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers/)._")
st.sidebar.markdown("Explore, predict churn rate and create clusters of customers")

# 1st part of the dashboard
st.sidebar.title("Explore")
select1 = st.sidebar.selectbox('Select a type of feature or the target',
                ['Target', 'Social features', 'Financial features', 'Numerical features', 'Bivariate'], key='1')

if not st.sidebar.checkbox("Hide", True, key='2'):
    st.markdown("### 1. Descriptive statistics")

    if select1 == 'Target':
        fig = piecharts(subtype=0)
    elif select1 == 'Social features':
        fig = piecharts(subtype=1)
    elif select1 == 'Financial features':
        fig = piecharts(subtype=2)
    elif select1 == 'Numerical features':
        varlist = histos()[1]
        select1a = st.selectbox('Select a numerical feature', varlist, key='3')
        fig = histos(select1a)[0]
    elif select1 == 'Bivariate':
        varlist1 = scatters()[1]
        select1b = st.selectbox('Select a feature', varlist1, key='4')
        select1c = st.selectbox('Select another feature', varlist1, key='5')
        fig = scatters(select1b, select1c)[0]
    st.pyplot(fig)


#  2nd part : classification for churn prediction
st.sidebar.title("Churn prediction")
select2 = st.sidebar.slider('Select number of trees to build in the random forest model', 100, 10, key='6')

if not st.sidebar.checkbox("Hide", True, key='7'):
    st.markdown("### 2. Churn prediction")
    st.markdown("Supervised classification evaluation using a Random Forest model")
    if select2:
        score, score_, report, confus_matrix, fig, fig2 = classify(model=RandomForestClassifier(n_estimators=select2, max_features=None))
        st.write('Score for train data:\n ', round(score,2))
        st.write('Score for test data:\n ', round(score_,2))

        st.write('Classification report:\n ')
        st.text('>\n ' + report)
        st.write('\n ')
        st.write('\n ')
        st.write('Confustion matrix:\n ')
        st.table(confus_matrix)
        st.write('\n ')
        st.write('\n ')
        st.pyplot(fig)
        st.write('\n ')
        st.write('\n ')
        st.pyplot(fig2)


#  3rd part : clustering strategy
st.sidebar.title("Clustering for a marketing strategy")
select3 = st.sidebar.slider('Select the upper bound of estimated probability to be attrited (in %):', 100, 20)
select4 = st.sidebar.slider('Now select the lower bound:', 0, select3)

if not st.sidebar.checkbox("Hide", True):
    st.markdown("### 3. Clustering for a marketing strategy")
    st.markdown("See how many customers are in the probability interval.")
    if select3:
        fig, a, b = probacluster(model=RandomForestClassifier(n_estimators=50, max_features=None), up=select3, lo=select4)
        st.write('Number of selected customers:\n ', a)
        st.write('Proportion:\n ', b)
        st.write('\n ')
        st.pyplot(fig)
