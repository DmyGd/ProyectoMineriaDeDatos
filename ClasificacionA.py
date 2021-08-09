import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Para la visualización de datos basado en matplotlib
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)


class ClasificationA():
    def __init__(self, data):
        self.data = data
        self.valor = ""
        self.X = ""
        self.Y = ""

    def algorithm_description(self):
        st.markdown("""
        <b>Proposito:</b>
        <p style="text-align: justify;"> 
        Predice etiquetas de una o más clases de tipo discretas (0, 1, 2) o nominales (A, B, C; o
        positivo, negativo; y otros). La regresión logística es otro tipo de algoritmo de aprendizaje supervisado cuyo objetivo es
        predecir valores binarios (0 o 1). Este algoritmo consiste en una transformación a la regresión
        lineal.
        </p>
        """,  unsafe_allow_html=True)

    # Acceso a datos y selección de características
    def feature_selection(self):
        st.write("### **Selección de caracteristicas **")
        if st.checkbox("Mostrar", key="selectClassif"):
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
            if st.checkbox("Generar graficos de dispersion, Matriz de correlación y mapa de calor (evaluación visual)", key="GrafDispClasif"):
                try:
                    self.valor = st.text_input(
                        "Ingresa el Hue para trabajar")
                    if len(self.valor) != 0:
                        # hue='Diagnosis'
                        sns.scatterplot(x='Radius', y='Perimeter',
                                        data=self.data, hue=self.valor)
                        plt.title('Gráfico de dispersión')
                        plt.xlabel('Radius')
                        plt.ylabel('Perimeter')
                        st.pyplot()

                        sns.scatterplot(
                            x='Concavity', y='ConcavePoints', data=self.data, hue=self.valor)
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

    def d_predictor_and_class_var(self):
        st.write("### **Definición de variables predictoras y variable clase **")
        if st.checkbox("Mostrar", key="selectdpfClassif"):
            self.data = self.data.replace({'M': 0, 'B': 1})
            try:
                self.valor = st.text_input("Ingresa el Hue para trabajar")
                if len(self.valor) != 0:
                    st.write(self.data.groupby(self.valor).size())

                    valores = st.text_input(
                        "Ingresa los valores en los que estas interezado seprados por una coma (Variables predictoras)", key="varSelectedClassif")
                    valoresSeleccionados = valores.replace(
                        " ", "").split(',')
                    print(valoresSeleccionados)
                    #['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension']
                    #Texture, Area, Smoothness, Compactness, Symmetry, FractalDimension
                    # Variables predictoras
                    self.X = np.array(self.data[valoresSeleccionados])
                    # X = BCancer.iloc[:, [3, 5, 6, 7, 10, 11]].values  #iloc para seleccionar filas y columnas según su posición
                    st.write("Varibles predictoras")
                    st.write(pd.DataFrame(self.X))
                    st.write("Variable Clase")
                    self.Y = np.array(self.data[[self.valor]])
                    st.write(pd.DataFrame(self.Y))

                    plt.figure(figsize=(10, 7))
                    #plt.scatter(X[:,0], X[:,1], c = self.data.Diagnosis)
                    plt.scatter(self.X[:, 0], self.X[:, 1],
                                c=self.data[self.valor])
                    plt.grid()
                    plt.xlabel('Texture')
                    plt.ylabel('Area')
                    st.pyplot()
            except:
                st.write("No se ha ingresado un parametro valido")

    def algorithm_application(self):
        st.write("### **Aplicación del algoritmo **")
        if st.checkbox("Mostrar", key="selectApClassif"):
            # Se declara el modelo de tipo regresión logística
            Clasificacion = linear_model.LogisticRegression()
            seed = 1234
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
                self.X, self.Y, test_size=0.2, random_state=seed, shuffle=True)
            st.write("X")
            st.write(pd.DataFrame(X_train))
            st.write("Y")
            st.write(pd.DataFrame(Y_train))
            st.write("Entrenando modelo...")
            # Se entrena el modelo a partir de los datos de entrada
            Clasificacion.fit(X_train, Y_train)
            st.write("Predicciones probabilisticas")
            # Predicciones probabilísticas
            Probabilidad = Clasificacion.predict_proba(X_train)
            st.write(pd.DataFrame(Probabilidad))
            # Predicciones con clasificación final
            st.write("#Predicciones con clasificación final ")
            Predicciones = Clasificacion.predict(X_train)
            st.write(pd.DataFrame(Predicciones))
            st.write("Evaluación de exactitud (Accuracy)")
            # Para la evaluación la exactitud (accuracy) se puede usar la función score()
            st.write(Clasificacion.score(X_train, Y_train))
            st.write("## Validando modelo")
            # Matriz de clasificación
            PrediccionesNuevas = Clasificacion.predict(X_validation)
            confusion_matrix = pd.crosstab(Y_validation.ravel(), PrediccionesNuevas, rownames=[
                                           'Real'], colnames=['Clasificación'])
            st.write("MAtriz de clasificación")
            st.write(confusion_matrix)
            # Reporte de la clasificación
            st.write("Exactitud:", Clasificacion.score(
                X_validation, Y_validation))
            st.write(classification_report(Y_validation, PrediccionesNuevas))
            # Ecuación del modelo
            st.write("Intercept:", Clasificacion.intercept_)
            st.write('Coeficientes: \n', Clasificacion.coef_)
