import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib


st.write(""" 
# TUGAS PENDAT
""")

description, importdata, preprocessing, modelling, implementation = st.tabs(["Description", "Import Data", "Preprocessing", "Modelling", "Implementation"])



with importdata:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)


with preprocessing:
    st.write("""# Preprocessing""")
    df = pd.read_csv("https://raw.githubusercontent.com/pramdf042/datamining/main/water_potability.csv")
    df['Sulfate'] = df['Sulfate'].fillna(value=df['Sulfate'].median())
    df['ph'] = df['ph'].fillna(value=df['ph'].median())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(value=df['Trihalomethanes'].median())
    df
    df[["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]].agg(['min','max'])
    df.Potability.value_counts()
    X = df.drop(columns="Potability")
    y = df["Potability"]
    "### Normalize data transformasi"
    X

    X.shape, y.shape

    #le.inverse_transform(y)

    labels = pd.get_dummies(df.Potability).columns.values.tolist()
    
    "### Label"
    labels

    # """## Normalisasi MinMax Scaler"""


    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    X

    X.shape, y.shape



with modelling:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # X_train = cv.fit_transform(X_train)
    # X_test = cv.fit_transform(X_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")

    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)



with implementation:
    st.write("# Implementation")
    ph = st.number_input('Masukkan pH Air')
    Hardness = st.number_input('Masukkan Hardness Air')
    Solids = st.number_input('Masukkan Solids Air')
    Chloramines = st.number_input('Masukkan Chloramines Air')
    Sulfate = st.number_input('Masukkan Sulfate Air')
    Conductivity = st.number_input('Masukkan Conductivity Air')
    Organic_carbon = st.number_input('Masukkan Organic Carbon Air')
    Trihalomethanes = st.number_input('Masukkan Trihalomethanes Air')
    Turbidity = st.number_input('Masukkan Turbidity Air')

    def submit():
        # input
        inputs = np.array([[
            ph,
            Hardness,
            Solids,
            Chloramines,
            Sulfate,
            Conductivity,
            Organic_carbon,
            Trihalomethanes,
            Turbidity
            ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()

