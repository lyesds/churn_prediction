import streamlit as st
from utils.b_descriptive import piecharts, histos

st.title("Bank customers churn prediction and clustering")
st.sidebar.title("Bank customers classification")
st.markdown("Welcome to this dashboard for bank customers insights!")
st.sidebar.markdown("Explore, predict churn rate and create clusters of customers")

st.sidebar.title("Explore")
select1 = st.sidebar.selectbox('1.Descriptive statistics',
                               ['Target', 'Social features', 'Financial features', 'Numerical features'], key='1')

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
        select1_ = st.selectbox('Numerical features', varlist, key='3')
        fig = histos(select1_)[0]
    st.pyplot(fig)
