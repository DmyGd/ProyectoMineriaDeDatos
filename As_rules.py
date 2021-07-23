import streamlit as st
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
# Para la generación de gráficas a partir de los datos
import matplotlib.pyplot as plt
from apyori import apriori
import json


class As_rules():
    def __init__(self, data):
        self.data = data

    # Regresa la descripción del algoritmo
    def algorithm_description(self):
        st.write("""**Proposito:** Consiste en identificar un conjunto de patrones secuenciales en forma de reglas de tipo:
            **A --> B**, con el fin de aumentar las ventas y reducir costos.\n
            Ejemplo: 
            \t * Artículos que se compran juntos con frecuencia
            \t * Síntomas asociados a un diagnóstico
            Usos: 
            \t * Colocación de productos
            \t * Publicidad Dirigida
            \t * Ventas
            \t * Cupones  
        """)

    def data_processing(self):
        st.write("### **Procesamiento de Datos**")
        if st.checkbox("Mostrar", key="AS"):
            dataList = []
            # for i in range(0, 7460):
            for i in range(0, 2):
                # print(i)
                dataList.append([str(self.data.values[i, j])
                                 for j in range(0, 20)])

            # st.write(dataList)
            if st.checkbox("Configuración 1"):
                ListaReglas1 = apriori(
                    dataList, min_support=0.01, min_confidence=0.3, min_lift=2)
                ReglasAsociacion1 = list(ListaReglas1)
                st.write("Número de reglas de asociación: ")
                st.write(len(ReglasAsociacion1))
                if st.checkbox("Motrar Todas las reglas de asociación", key="mostrarReglasC1"):
                    st.write(ReglasAsociacion1)
                if st.checkbox("Motrar reglaespecifica", key="mostrarSpecificReglasC1"):
                    valor = st.number_input(
                        "Ingresa el numero de regla a mostrar", min_value=1, key="inputAS_RulCAC1")
                    print(valor)
                    if valor < 1 or valor > len(ReglasAsociacion1):
                        st.write("*Numero fuera de rango*")
                    else:
                        st.write(
                            f" Regla [{valor}] \n{ReglasAsociacion1[valor-1]}")

            if st.checkbox("Configuración 2"):
                ListaReglas2 = apriori(
                    dataList, min_support=0.028, min_confidence=0.3, min_lift=1.01)
                ReglasAsociacion2 = list(ListaReglas2)
                st.write("Número de reglas de asociación: ")
                st.write(len(ReglasAsociacion2))
                if st.checkbox("Motrar Todas las reglas de asociación", key="mostrarReglasC2"):
                    st.write(ReglasAsociacion2)
                if st.checkbox("Motrar reglaespecifica", key="mostrarSpecificReglasC2"):
                    valor = st.number_input(
                        "Ingresa el numero de regla a mostrar", min_value=1, key="inputAS_RulCAC2")
                    print(valor)
                    if valor < 1 or valor > len(ReglasAsociacion2):
                        st.write("*Numero fuera de rango*")
                    else:
                        st.write(
                            f" Regla [{valor}] \n{ReglasAsociacion2[valor-1]}")


if __name__ == "__main__":
    print("Funcionas?")
