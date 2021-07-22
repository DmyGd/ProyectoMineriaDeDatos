import streamlit as st
import pandas as pd  # Para la manipulación y análisis de datos
import numpy as np  # Para crear vectores y matrices n dimensionales
# Para la generación de gráficas a partir de los datos
import matplotlib.pyplot as plt
import seaborn as sns  # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PCA_MD():
    def __init__(self, data):
        self.data = data

    # Mdata es una matriz con los valores dropeados
    def matriz_nomalizada(self, Mdata):
        normalizar = StandardScaler()  # Se instancia el objeto StandardScaler
        # Se calcula la media y desviación para cada dimensión
        normalizar.fit(Mdata)
        # Se normalizan los datos
        MNormalizada = normalizar.transform(Mdata)
        return MNormalizada

    # Recibe una Matriz normalizada y el numero de componentes, si no
    # se especifica el numero de componentes, por default es 9
    def get_component(self, MNormalizada, num_comp=9):
        Componentes = PCA(n_components=num_comp)
        # Se obtiene los componentes
        Componentes.fit(MNormalizada)
        return Componentes

    # Regresa la descripción del algoritmo
    def algorithm_description(self):
        st.write("""**Proposito:** es un algoritmo
        matemático para reducir la cantidad de variables de conjuntos de datos, mientras se conserva la mayor
        cantidad de información posible.""")

    def data_standardization(self):
        st.write("### **Estandarización de los datos**")
        if st.checkbox("Mostrar", key="estandarizaciónPCA"):
            # normalizar = StandardScaler()  # Se instancia el objeto StandardScaler
            try:
                valores = st.text_input(
                    "Ingresa el o los valores que desea retirar (debe ingresar minimo un valor)", key="inputEstandarDataPCA")
                valoresDrop = valores.replace(" ", "").split(',')
                print(valoresDrop)
                if len(valoresDrop) != 0:
                    #Mdata = self.data.drop(['comprar'], axis=1)
                    Mdata = self.data.drop(valoresDrop, axis=1)
                    MNormalizada = self.matriz_nomalizada(Mdata)
                    st.write(f"Valores dependientes eliminados: {valoresDrop}")
                    st.write(MNormalizada.shape)
                    st.write("Matriz de datos normalizada: ")
                    st.write(pd.DataFrame(MNormalizada, columns=Mdata.columns))
            except:
                st.write(
                    "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")

    def cov_components_varianza(self):
        st.write("### **Calculo de matriz de Covarianza, Componentes y Varianza**")
        if st.checkbox("Mostrar", key="cov_components_varianzaPCA"):
            try:
                valores = st.text_input(
                    "Ingresa el o los valores que desea retirar (debe ingresar minimo un valor)", key="inputCovComVarPCA")
                valoresDrop = valores.replace(" ", "").split(',')
                # print(valoresDrop)
                if len(valoresDrop) != 0:
                    #Mdata = self.data.drop(['comprar'], axis=1)
                    Mdata = self.data.drop(valoresDrop, axis=1)
                    MNormalizada = self.matriz_nomalizada(Mdata)
                    Componentes = self.get_component(MNormalizada)
                    #X_Comp = Componentes.transform(MNormalizada)
                    # pd.DataFrame(X_Comp)
                    st.write(f"Valores dependientes eliminados: {valoresDrop}")
                    st.write(Componentes.components_)
            except:
                st.write(
                    "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")

    def main_components(self):
        st.write("### **Componentes principales**")
        if st.checkbox("Mostrar", key="mainCompPCA"):
            try:
                valores = st.text_input(
                    "Ingresa el o los valores que desea retirar (debe ingresar minimo un valor)", key="inputMainCompPCA")
                valoresDrop = valores.replace(" ", "").split(',')
                print(valoresDrop)
                if len(valoresDrop) != 0:
                    #Mdata = self.data.drop(['comprar'], axis=1)
                    Mdata = self.data.drop(valoresDrop, axis=1)
                    #MNormalizada = self.matriz_nomalizada(Mdata)
                    Componentes = self.get_component(
                        self.matriz_nomalizada(Mdata))
                    #X_Comp = Componentes.transform(MNormalizada)
                    # pd.DataFrame(X_Comp)
                    st.write(f"Valores dependientes eliminados: {valoresDrop}")
                    # st.write(Componentes.components_)
                    Varianza = Componentes.explained_variance_ratio_
                    st.write('Eigenvalues:')
                    st.write(Varianza)
                    st.write('Varianza acumulada:')
                    #############################
                    st.write(sum(Varianza[0:5]))
                    #############################
                    # Con 5 componentes se tiene el 85% de varianza acumulada y con 6 el 91%
                    st.write(
                        "Grafica de Varianza acumulada en las nuevas dimenciones")
                    # Se grafica la varianza acumulada en las nuevas dimensiones
                    plt.plot(np.cumsum(Componentes.explained_variance_ratio_))
                    plt.xlabel('Número de componentes')
                    plt.ylabel('Varianza acumulada')
                    plt.grid()
                    st.pyplot()
            except:
                st.write(
                    "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")

    def relevancy_ratio_cargas_(self):
        st.write("### **Proporción de relevancias –cargas–**")
        if st.checkbox("Mostrar", key="cargasPCA"):
            try:
                valores = st.text_input(
                    "Ingresa el o los valores que desea retirar (debe ingresar minimo un valor)", key="inputCargasCA")
                valoresDrop = valores.replace(" ", "").split(',')
                print(valoresDrop)
                if len(valoresDrop) != 0:
                    #Mdata = self.data.drop(['comprar'], axis=1)
                    Mdata = self.data.drop(valoresDrop, axis=1)
                    #MNormalizada = self.matriz_nomalizada(Mdata)
                    Componentes = self.get_component(
                        self.matriz_nomalizada(Mdata))
                    st.write(pd.DataFrame(abs(Componentes.components_)))

                    CargasComponentes = pd.DataFrame(
                        Componentes.components_, columns=Mdata.columns)
                    st.write(CargasComponentes)

                    CargasComponentes = pd.DataFrame(
                        abs(Componentes.components_), columns=Mdata.columns)
                    st.write(CargasComponentes)
            except:
                st.write(
                    "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")

    def create_new_matrix(self):
        st.write("### **Generar una matriz nueva**")
        if st.checkbox("Mostrar", key="matrizPCA"):
            try:
                valores = st.text_input(
                    "Ingresa el o los valores que desea retirar (debe ingresar minimo un valor)", key="inputCargasCA")
                valoresDrop = valores.replace(" ", "").split(',')
                print(valoresDrop)
                if len(valoresDrop) != 0:
                    #Mdata = self.data.drop(['comprar'], axis=1)
                    #['ahorros', 'vivienda', 'estado_civil', 'hijos']
                    Mdata = self.data.drop(valoresDrop, axis=1)
                    st.write(Mdata)
            except:
                st.write(
                    "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")


if __name__ == "__main__":
    print("Funcionas?")
