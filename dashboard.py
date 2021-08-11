import streamlit as st

# import churn_prediction.utils.c_supervised_classif
from utils.b_descriptive import piecharts, histos, scatters
from utils.c_supervised_classif import classify
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

plt.style.use('seaborn-muted')


st.title("Bank customers churn prediction and clustering")
st.sidebar.title("Bank customers classification")
st.markdown("Welcome to this dashboard for bank customers insights!")
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
        score, score_, report, confus_matrix, fig = classify(model=RandomForestClassifier(n_estimators=select2, max_features=None))
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
