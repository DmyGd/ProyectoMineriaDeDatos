import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D


st.set_option('deprecation.showPyplotGlobalUse', False)


class Kmean_md():
    def __init__(self, data):
        self.data = data
        self.MatrizVariables = []
        self.valoresSeleccionados = []

    def algorithm_description(self):
        st.markdown("""
        <b>Proposito:</b>
        <p style="text-align: justify;"> 
        Este algoritmo es parte del aprendizaje no supervisado, en este el objetivo es dividir 
        una población heterogénea de elementos en un número de grupos naturales (regiones o segmentos homogéneos)
        de acuerdo a sus similitudes.
        </p>
        """,  unsafe_allow_html=True)

    def feature_selection(self):
        st.write("### **Selección de caracteristicas **")
        if st.checkbox("Mostrar", key="selectKmeans"):
            if st.checkbox("Generar Matriz de correlaciones (evaluación visual)", key="MatrixCorrKmeans"):
                try:
                    valor = st.text_input(
                        "Ingresa el Hue para trabajar")
                    if len(valor) != 0:
                        #sns.pairplot(self.data, hue='Diagnosis')
                        sns.pairplot(self.data, hue=valor)
                        st.pyplot()
                except:
                    st.write("No se ha ingresado un parametro valido")
            if st.checkbox("Generar graficos de dispersion, Matriz de correlación y mapa de calor (evaluación visual)", key="GrafDispKmeans"):
                try:
                    valor = st.text_input(
                        "Ingresa el Hue para trabajar")
                    if len(valor) != 0:
                        # hue='Diagnosis'
                        sns.scatterplot(x='Radius', y='Perimeter',
                                        data=self.data, hue=valor)
                        plt.title('Gráfico de dispersión')
                        plt.xlabel('Radius')
                        plt.ylabel('Perimeter')
                        st.pyplot()

                        sns.scatterplot(
                            x='Concavity', y='ConcavePoints', data=self.data, hue=valor)
                        plt.title('Gráfico de dispersión')
                        plt.xlabel('Concavity')
                        plt.ylabel('ConcavePoints')
                        st.pyplot()

                        st.write(
                            "Matriz de correlaciones con Metodo **Person**")
                        CorrData = self.data.corr(method='pearson')
                        st.write(CorrData)

                        st.write("Mapa de calor")
                        plt.figure(figsize=(14, 7))
                        MatrizInf = np.triu(CorrData)
                        sns.heatmap(CorrData, cmap='RdBu_r',
                                    annot=True, mask=MatrizInf)
                        st.pyplot()

                        st.write("Top 10 valores")
                        st.write(CorrData['Radius'].sort_values(
                            ascending=False)[:10])
                except:
                    st.write("No se ha ingresado un parametro valido")
            if st.checkbox("Selección Variables", key="VarsSelectedCorrKmeans"):
                try:
                    valores = st.text_input(
                        "Ingresa los valores en los que estas interezado seprados por una coma", key="varSelectedKmeans")
                    self.valoresSeleccionados = valores.replace(
                        " ", "").split(',')
                    # ['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'
                    #Texture, Area, Smoothness, Compactness, Symmetry, FractalDimension
                    print(self.valoresSeleccionados)
                    if len(self.valoresSeleccionados) != 0:
                        self.MatrizVariables = np.array(
                            self.data[self.valoresSeleccionados])
                        st.write(pd.DataFrame(self.MatrizVariables))

                except:
                    st.write(
                        "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")

    def kmeans_algorithm(self):
        st.write("### **Aplicar algoritmo Kmeans**")
        st.write(
            "#### **Nota:** Debio haberse realizado previamente la selección de variables en:\n *Selección de caracteristicas > Slección de variables*")
        if st.checkbox('Mostrar', key="DeteccionValoresEDA"):
            if len(self.MatrizVariables) == 0:
                st.write("No ha realizado una selección de variables")
            else:
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0)
                    km.fit(self.MatrizVariables)
                    SSE.append(km.inertia_)

                # Se grafica SSE en función de k
                plt.figure(figsize=(10, 7))
                plt.plot(range(2, 12), SSE, marker='o')
                plt.xlabel('Cantidad de clusters *k*')
                plt.ylabel('SSE')
                plt.title('Elbow Method')
                st.write("Elbow Method")
                st.pyplot()
                st.write("Knee")
                kl = KneeLocator(range(2, 12), SSE,
                                 curve="convex", direction="decreasing")
                kl.elbow
                plt.style.use('ggplot')
                fig = kl.plot_knee()
                st.pyplot(fig)

                # Se crean las etiquetas de los elementos en los clusters
                st.write("Creando etiquetas de los elementos en los clusters")
                MParticional = KMeans(n_clusters=5, random_state=0).fit(
                    self.MatrizVariables)
                MParticional.predict(self.MatrizVariables)
                # st.write(f"{MParticional.labels_}")
                self.data['clusterP'] = MParticional.labels_
                st.write(self.data)
                st.write("Conteo:")
                st.write(self.data.groupby(['clusterP'])['clusterP'].count())
                st.write("Grafica cluster 2D")
                plt.figure(figsize=(10, 7))
                plt.scatter(
                    self.MatrizVariables[:, 0], self.MatrizVariables[:, 1], c=MParticional.labels_, cmap='rainbow')
                st.pyplot()
                st.write(
                    "Gráfica de los elementos y los centros de los clusters (3D)")
                CentroidesP = MParticional.cluster_centers_
                pd.DataFrame(CentroidesP.round(
                    4), columns=self.valoresSeleccionados)
                # st.write(pd.DataFrame(CentroidesP.round(
                #    4), columns=self.valoresSeleccionados))
                # Gráfica de los elementos y los centros de los clusters

                plt.rcParams['figure.figsize'] = (10, 7)
                plt.style.use('ggplot')
                colores = ['red', 'blue', 'cyan', 'green', 'yellow']
                asignar = []
                for row in MParticional.labels_:
                    asignar.append(colores[row])

                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(self.MatrizVariables[:, 0], self.MatrizVariables[:, 1],
                           self.MatrizVariables[:, 2], marker='o', c=asignar, s=60)
                ax.scatter(CentroidesP[:, 0], CentroidesP[:, 1],
                           CentroidesP[:, 2], marker='o', c=colores, s=1000)
                st.pyplot()

                if st.checkbox("Consultar valores cercanos al centroide"):
                    # Es posible identificar los pacientes más cercanos a cada centroide
                    Cercanos, _ = pairwise_distances_argmin_min(
                        MParticional.cluster_centers_, self.MatrizVariables)
                    st.write(Cercanos)

                    try:
                        # IDNumber
                        columna = st.text_input(
                            "Ingrese el nombre de la columna que desea consultar", key="CercanosdKmeans")
                        if len(columna) != 0:
                            report = self.data[columna].values
                            for row in Cercanos:
                                st.write(report[row])
                    except:
                        st.write(
                            "No se ha ingresdo ningun parametro o el parametro ingreado no es valido")


if __name__ == "__main__":
    print("Funcionas?")
