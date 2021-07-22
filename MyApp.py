import streamlit as st
import pandas as pd  # Para la manipulacion y análisis de datos
import numpy as np  # Para crear vectores de datos de n dimenciones
import matplotlib.pyplot as plt  # Para generar gráficos
import seaborn as sns  # Para visualización de los datos
from EDA import EDA

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
    ('EDA', 'Algoritmo 2', 'Algoritmo 3')
)
st.write("""
# Estas en el algoritmo: 
""")
st.write(f"## *{add_selectbox}*")


if add_selectbox == 'EDA':
    try:
        eda = EDA(data)
        st.write(f"{eda.algorithm_description()}")
        eda.description_data_structure()
        eda.missing_data()
        eda.outlier_detection()
        eda.relations_variables()

    except:
        st.write("No se ha ingresado una fuente de información")
