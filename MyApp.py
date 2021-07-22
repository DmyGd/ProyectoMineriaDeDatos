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
        # st.write(f"{eda.print_data()}")
        if st.checkbox('Mostrar todos los datos'):
            st.write(data)
        if st.checkbox('Mostrar cabecera'):
            st.write(data.head())
        st.write("### **Descripción de la estructura de datos**")
        if st.checkbox('Mostrar', key="DescripciónEDA"):
            eda.description_data_structure()

        st.write("### **Identificación de datos faltantes**")
        if st.checkbox('Mostrar', key="IdentificaciónEDA"):
            eda.missing_data()

        st.write("### **Deteccion de valores Atipicos**")
        if st.checkbox('Mostrar', key="DeteccionValoresEDA"):
            eda.outlier_detection()

        st.write("### **Identificación de relaciónes entre variables**")
        if st.checkbox("Mostrar", key="IdentificacionRelEDA"):
            eda.relations_variables()

    except:
        st.write("No se ha ingresado una fuente de información")
