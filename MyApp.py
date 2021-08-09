import streamlit as st
import pandas as pd  # Para la manipulacion y análisis de datos
import numpy as np  # Para crear vectores de datos de n dimenciones
import matplotlib.pyplot as plt  # Para generar gráficos
import seaborn as sns  # Para visualización de los datos
from EDA import EDA
from PCA_MD import PCA_MD
from As_rules import As_rules
from Kmean_md import Kmean_md
from ClasificacionA import ClasificationA
# comando para ejecutar: streamlit run file.py
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Mineria de Datos
*Autor: Dominguez Duran Gerardo* 
""")
file = st.file_uploader("Archivo", type=["csv"])
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
    ('EDA', 'PCA', 'Reglas de asociación', 'Cluster Particional',
     'Clasificación con Regresión Logística')
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

    if add_selectbox == 'Reglas de asociación':
        as_rules = As_rules(data)
        as_rules.algorithm_description()
        as_rules.data_processing()

    if add_selectbox == 'Cluster Particional':
        kmean = Kmean_md(data)
        kmean.algorithm_description()
        kmean.feature_selection()
        kmean.kmeans_algorithm()

    if add_selectbox == 'Clasificación con Regresión Logística':
        clasificacion = ClasificationA(data)
        clasificacion.algorithm_description()
        clasificacion.feature_selection()
        clasificacion.d_predictor_and_class_var()
        clasificacion.algorithm_application()
except:
    st.write("No se ha ingresado una fuente de información")
