import pandas as pd
import streamlit as st
import plotly.express as px



def main():
    st.title("Exploratory Data Analysis (EDA) with Plotly and Streamlit")


    # uploaded_file = st.file_uploader("data.csv", type="csv")
    
    df = pd.read_csv('data.csv')    
    st.subheader("DataFrame:")
    st.dataframe(df)

    st.subheader("Summary Statistics:")
    st.write(df.describe())

    st.subheader("Distributions of Numerical Columns:")
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object'])
    for column in numerical_columns:
        fig = px.histogram(df, x=column, title=f'Distribution of {column}')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()