import streamlit as st
import pandas as pd  # Para la manipulacion y análisis de datos
import numpy as np  # Para crear vectores de datos de n dimenciones
import matplotlib.pyplot as plt  # Para generar gráficos
import seaborn as sns  # Para visualización de los datos
import altair as alt
import io

st.set_option('deprecation.showPyplotGlobalUse', False)


class EDA():
    def __init__(self, data):
        self.data = data

    # Regresa la descripción del algoritmo
    def algorithm_description(self):
        des = """**Proposito:** Tener una idea de la estructura del conjunto de datos, identificar la variable objetivo y posibles técnicas de modelado."""
        return des

    def description_data_structure(self):
        st.write("### **Descripción de la estructura de datos**")
        if st.checkbox('Mostrar', key="DescripciónEDA"):
            st.write("Cantidad de Filas y columnas que tiene el conjunto de datos:")
            st.write(self.data.shape)
            #st.markdown('<h1>Prueba HTML</h1>',  unsafe_allow_html=True)
            st.write("Tipos de datos por variable")
            st.write(self.data.dtypes)

    def missing_data(self):
        st.write("### **Identificación de datos faltantes**")
        if st.checkbox('Mostrar', key="IdentificaciónEDA"):
            st.write("valores nulos en cada variable:")
            st.write(self.data.isnull().sum())
            # Debido a que la salida estandar es std.out
            #st.write("Tipo de Dato y suma de valores nulos: ")
            #buffer = io.StringIO()
            # self.data.info(buf=buffer)
            # st.write(buffer.getvalue())

    def outlier_detection(self):
        st.write("### **Deteccion de valores Atipicos**")
        if st.checkbox('Mostrar', key="DeteccionValoresEDA"):
            # if st.checkbox('#### 1.- Distribución de variables númericas', key="DistribucionVarEDA"):
            if st.checkbox("1. Generar graficas de distribución de variables numéricas"):
                #fig = plt.figure(self.data.hist(figsize=(14, 14), xrot=45))
                #fig = plt.figure()
                self.data.hist(figsize=(15, 15))
                st.pyplot()

            if st.checkbox("2. Generar resumen estadístico de variables numéricas"):
                st.write(self.data.describe())

            if st.checkbox('3. Generar diagramas para detectar posibles valores atípicos', key="DiagramasAtipicosEDA"):
                #['Price', 'Landsize','BuildingArea', 'YearBuilt']
                try:
                    valores = st.text_input(
                        "Ingresa los valores en los que estas interezado seprados por una coma")
                    valoresAtipicos = valores.replace(" ", "").split(',')
                    valoresAtipicos = valoresAtipicos
                    if len(valoresAtipicos) != 0:
                        print("***********")
                        print(valoresAtipicos)
                        for col in valoresAtipicos:
                            sns.boxplot(col, data=self.data)
                            st.pyplot()
                            valoresAtipicos = []
                            valores = ""
                except:
                    st.write(
                        "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")

            if st.checkbox("4. Mostrar la distribución de variables categóricas"):
                st.write((self.data.describe(include='object')))
                for col in self.data.select_dtypes(include='object'):
                    if self.data[col].nunique() < 10:
                        sns.countplot(y=col, data=self.data)
                        st.pyplot()
            if st.checkbox("5. Mostrar la agrupación por variables categóricas"):
                for col in self.data.select_dtypes(include='object'):
                    if self.data[col].nunique() < 10:
                        st.write((self.data.groupby(col).agg(['mean'])))

    def relations_variables(self):
        st.write("### **Identificación de relaciónes entre variables**")
        if st.checkbox("Mostrar", key="IdentificacionRelEDA"):
            if st.checkbox("Generar Matriz de correlación", key="MatrizCorrEDA"):
                st.write(self.data.corr())

            if st.checkbox("Mapa de calor", key="mapaCalorEDA"):
                fig = plt.figure(figsize=(14, 14))
                sns.heatmap(self.data.corr(), cmap='RdBu_r', annot=True)
                st.pyplot(fig)


if __name__ == "__main__":
    eda = EDA(pd.read_csv("../../MaterialMineria/melb_data.csv"))
    print(eda.algorithm_description())
    datos = pd.read_csv("../../MaterialMineria/melb_data.csv")
    datos.hist(figsize=(15, 15))
    st.pyplot()
