import streamlit as st
import pandas as pd  # Para la manipulacion y análisis de datos
import numpy as np  # Para crear vectores de datos de n dimenciones
import matplotlib.pyplot as plt  # Para generar gráficos
import seaborn as sns  # Para visualización de los datos
from EDA import EDA
from PCA_MD import PCA_MD
# comando para ejecutar: streamlit run file.py
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Mineria de Datos
*Autor: Dominguez Duran Gerardo* 
""")
file = st.file_uploader("Archivo", type="csv")
# print(file)

if file != None:
    data = pd.read_csv(file)
    if st.checkbox('Mostrar todos los datos'):
        st.write(data)
    if st.checkbox('Mostrar cabecera'):
        st.write(data.head())

# Barra de control lateral Izquierdo
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'Algoritmo Actual',
    ('EDA', 'PCA', 'Algoritmo 3')
)
st.write("""
# Estas en el algoritmo: 
""")
st.write(f"## *{add_selectbox}*")


try:
    if add_selectbox == 'EDA':
        eda = EDA(data)
        eda.algorithm_description()
        eda.description_data_structure()
        eda.missing_data()
        eda.outlier_detection()
        eda.relations_variables()
    if add_selectbox == 'PCA':
        pca_md = PCA_MD(data)
        pca_md.algorithm_description()
        pca_md.data_standardization()
        pca_md.cov_components_varianza()
        pca_md.main_components()
        pca_md.relevancy_ratio_cargas_()
        pca_md.create_new_matrix()
except:
    st.write("No se ha ingresado una fuente de información")
