import pandas as pd
import scipy
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import io
import numpy as np
from scipy.stats import norm
import streamlit as st

column_descriptions = {
    'GENDER': 'В данных встречается больше клиентов мужского пола',
    'CHILD_TOTAL': 'У большинства клиентов 0-2 детей',
    'EDUCATION': 'Cреднее и Среднее специальное образование встречается чаще всего, в то время как более высококвалифицированных специалистов с двумя и более высшими и учеными степенями почти нет',
    'CLOSED_FL': 'У 58% клиентов кредит на данный момент не закрыт',
    'GEN_INDUSTRY': 'Среди клиентов большинство имеют работу в сфере торговли',
    'GEN_TITLE': 'Более половины кредитов взято людям с должностью специалист',
    'JOB_DIR': 'Подавляющее большинство клиентов участвует в основной деятельности компании',
    'DEPENDANTS': 'Чаще всего у клиента нет иждивенцев',
    'SOCSTATUS_PENS_FL': '86.5% клиентов не являются пенсионерами',
    'FL_PRESENCE_FL': 'У 69% клиентов есть собственная квартира',
    'OWN_AUTO': 'У 88% есть одна машина, у 0.0125% клиентов есть 2 машины',
    'TERM': 'Самыми популярными сроками кредита являются 6-ти и 12-ти месячные кредиты ',
    'TARGET': 'Целевая переменная распределена не равномерно, более 85% не откликнулось на маркетинговую кампанию, что является нормой в задачах такого типа',
    'N_LOANS': 'Большинство клиентов брали только один кредит, есть',
    'REG_ADDRESS_PROVINCE': 'Кемеровская область и Приморский край -- самые популярные области регистрации клиентов'
}

def create_pie_chart(df, col):
    return px.pie(df[col].value_counts().reset_index(), values=col, names='index')

def create_bar_chart(df, col):
    return

def main():
    st.title("Exploratory Data Analysis (EDA)")
    
    df = pd.read_csv('data.csv')
    to_category = ['GENDER', 'CHILD_TOTAL','EDUCATION', 'CLOSED_FL', 'DEPENDANTS', 'SOCSTATUS_WORK_FL', 
                   'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'OWN_AUTO', 'N_LOANS', 'WORK_TIME']
    for col in to_category + list(df.select_dtypes('object').columns):
        df[col] = df[col].astype('category')
    target_column = 'TARGET'
    st.subheader("DataFrame:")
    st.dataframe(df)

    st.subheader("Summary Statistics:")
    st.write(df.describe())

    st.subheader("Columns")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()

    st.text(s)

    st.subheader("1. Распределение Numerical Data:")

    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns    
    selected_column = st.selectbox("Select a column:", df.select_dtypes(include=['int64', 'float64']).columns)

    if selected_column != 'AGE':
        q1 = df[selected_column].quantile(0.25)
        q3 = df[selected_column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(q1 - 3 * iqr, df[selected_column].min())
        upper_bound = min(3 + 3 * iqr, df[selected_column].max())
        fig = ff.create_distplot([df[selected_column].values], [selected_column], curve_type='normal', show_hist=False)
    else: 
        lower_bound = df[selected_column].min()
        upper_bound = df[selected_column].max()

        fig = ff.create_distplot([df[selected_column].values], [selected_column], curve_type='kde')
        # Compare with a normal distribution
        mean, std = df[selected_column].mean(), df[selected_column].std()
        x_values = np.linspace(df[selected_column].min(), df[selected_column].max(), 1000)
        y_values = norm.pdf(x_values, mean, std)
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Normal Distribution', line=dict(color='red')))

    fig.update_traces(marker_color='green', selector=dict(type='histogram'))
    fig.update_layout(xaxis_title=selected_column, yaxis_title='Density')
    fig.update_layout(title_text=f'Распределение {selected_column}')
    fig.update_layout(xaxis=dict(range=[lower_bound, upper_bound]))
    st.plotly_chart(fig)

    st.header("2. Распределение категориальных признаков и таргета")
    pies = ['SOCSTATUS_PENS_FL', 'FAMILY_INCOME', 'GENDER', 
            'OWN_AUTO', 'CLOSED_FL', 'FL_PRESENCE_FL']
    h_bars = ['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE','POSTAL_ADDRESS_PROVINCE',
              'GEN_INDUSTRY','GEN_TITLE','JOB_DIR']
    v_bars = ['EDUCATION', 'CHILD_TOTAL','DEPENDANTS', 
              'N_LOANS', 'TERM', 'TARGET', 'WORK_TIME']
    ordered_col = ['GENDER','EDUCATION','GEN_INDUSTRY',
                   'GEN_TITLE','JOB_DIR','WORK_TIME', 
                   'N_LOANS', 'TERM', 'CLOSED_FL','FAMILY_INCOME',
                   'MARITAL_STATUS','CHILD_TOTAL','DEPENDANTS',
                   'SOCSTATUS_PENS_FL','FL_PRESENCE_FL', 'OWN_AUTO',
                   'REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE','POSTAL_ADDRESS_PROVINCE',
                   'TARGET']
    for col in ordered_col:
        if col in pies:
            fig = px.pie(df, names=col)
            fig.update_traces(textinfo='label+percent', hoverinfo='label+percent+value')
        if col in h_bars:
            counts = df[col].value_counts().sort_values(ascending=True).reset_index()
            fig = px.bar(counts, x='count', y=col, orientation='h')
            fig.update_layout(xaxis_title='Counts', yaxis_title=None)
        if col in v_bars:
            counts = df[col].value_counts().sort_values(ascending=False).reset_index()
            fig = px.bar(counts, x=col, y='count')
            fig.update_layout(xaxis_title='Counts', yaxis_title=None)
        fig.update_layout(title_text=f'Распределение признака {col}')    
        st.plotly_chart(fig)  
        if col in  ['TERM',]:
            fig = px.box(df, x=target_column, y=col, points="all", title=f'Зависимость между {col} и {target_column}')
            st.plotly_chart(fig)
        if col in column_descriptions:
            st.write(column_descriptions[col])

    st.header("3. Correlation Matrix")
    correlation_matrix = df.corr(numeric_only=True)

    # Create an annotated heatmap with Plotly Express
    fig_corr = px.imshow(correlation_matrix,
                         labels=dict(x="Columns", y="Columns", color="Correlation"),
                         x=correlation_matrix.columns,
                         y=correlation_matrix.columns,
                         color_continuous_scale='Viridis')
    st.plotly_chart(fig_corr)
    st.write("""Наибольшая корреляция между """)

    st.header("4. Зависимость между таргетом и признаками")
    for column in numerical_columns:
        if column != target_column:
            fig = px.box(df, x=target_column, y=column, points="all", title=f'Зависимость между {column} и {target_column}')
            st.plotly_chart(fig)


if __name__ == '__main__':
    main()