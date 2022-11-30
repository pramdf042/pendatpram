import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import os
import warnings
import altair as alt
from sklearn.utils.validation import joblib


st.write(""" 
# TUGAS PENDAT
""")

description, importdata, preprocessing, modelling, implementation = st.tabs(["Description", "Import Data", "Preprocessing", "Modelling", "Implementation"])
warnings.filterwarnings("ignore")
with importdata:
    st.write("""# Upload File""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)

with preprocessing:
    prepros = st.radio(
    "Silahkan pilih metode preprocessing :",
    ("Normalisasi", "Min Max Scaler", "Categorical to Numeric"))
    prepoc = st.button("Preprocessing")

    if prepros == "Min Max Scaler":
        if prepoc:
            df[["Area", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Perimeter", "Roundness", "AspectRation"]].agg(['min','max'])
            df.Class.value_counts()
            X = df.drop(columns=["Class","id"])
            y = df["Class"]
            "### Normalize data transformasi"
            X
            X.shape, y.shape
            #   le.inverse_transform(y)
            labels = pd.get_dummies(df.Class).columns.values.tolist()
            "### Label"
            labels
            # """## Normalisasi MinMax Scaler"""
            scaler = MinMaxScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            X
            X.shape, y.shape




with modelling:
    # df = pd.read_csv("https://raw.githubusercontent.com/pramdf042/datamining/main/riceClassification.csv")
    X=df.iloc[:,:-1]
    y=df.iloc[:,-1]
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
    

with implementation:
    st.write("# Implementation")
    Area = st.number_input('Masukkan Area')
    MajorAxisLength = st.number_input('Masukkan MajorAxisLength')
    MinorAxisLength = st.number_input('Masukkan MinorAxisLength')
    Eccentricity = st.number_input('Masukkan Eccentricity')
    ConvexArea = st.number_input('Masukkan ConvexArea')
    EquivDiameter = st.number_input('Masukkan EquivDiameter')
    Extent = st.number_input('Masukkan Extent')
    Perimeter = st.number_input('Masukkan Perimeter')
    Roundness = st.number_input('Masukkan Roundness')
    AspectRation = st.number_input('Masukkan AspectRation')

    def submit():
        # input
        inputs = np.array([[
            Area, MajorAxisLength, MinorAxisLength, Eccentricity, ConvexArea, EquivDiameter, Extent, Perimeter, Roundness, AspectRation
        ]])
        # st.write(inputs)
        # baru = pd.DataFrame(inputs)
        # input = pd.get_dummies(baru)
        # st.write(input)
        # inputan = np.array(input)
        # import label encoder
        le = joblib.load("le.save")
        model1 = joblib.load("tree.joblib")
        y_pred3 = model1.predict(inputs)
        if le.inverse_transform(y_pred3)[0]==1:
            hasilakhir='Jasmine'
        else :
            hasilakhir='Gonen'
        st.write(f"Berdasarkan data yang Anda masukkan, maka beras dinyatakan memiliki jenis: {hasilakhir}")

    all = st.button("Submit")
    if all :
        submit()
