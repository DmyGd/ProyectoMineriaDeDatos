# Proyecto de Minería de Datos
## Interfaz de usuario (Documentación de código)
**Autor: Domínguez Duran Gerardo** 

Este proyecto tiene la intención de proporcionar una interfaz de usuario con cuatro algoritmos importantes en la Minería de Datos. Los algoritmos trabajados en este proyecto son: 
1. Análisis Exploratorio de Datos (EDA).
2. Análisis de componentes principales (PCA).
3. Clustering Particional.
4. Clasicación con Regresión logística.

En este documento se explicará de manera técnica partes relevantes del mismo.

Las tecnologías y herramientas utilizadas en este proyecto fueron:
* Entorno de desarrollo utilizado: **Visual Studio Code**, es un editor de código fuente desarrollado por Microsoft para Windows, Linux y macOS. Incluye soporte para la depuración, control integrado de Git, resaltado de sintaxis, finalización inteligente de código, fragmentos y refactorización de código. Disponible en [https://code.visualstudio.com/download](https://code.visualstudio.com/download) 
* Pluggins utilizados: **Prettier**, es un formateador de código. Aplica un estilo coherente al analizar su código y volver a imprimirlo con sus propias reglas que tienen en cuenta la longitud máxima de línea, ajustando el código cuando es necesario. Disponible en [https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)
* Librería utilizada: **Streamlit**, convierte los scripts de datos en aplicaciones web que se pueden compartir en minutos. No se requiere experiencia en front-end. Disponible en [https://streamlit.io/](https://streamlit.io/)
* Controlador de versiones: **GitHub**, es una herramienta comúnmente utilizada para el versiona miento de proyectos y a su vez alojar el código de las aplicaciones de cualquier desarrollador. Disponible en [https://github.com/](https://github.com/)
* Lenguaje de programación utilizado: **Python 3**, permite trabajar rápidamentee integrar los sistemas de forma más eficaz. Su interpreta esta disponible en [https://www.python.org/](https://www.python.org/)

## Proyecto
Se puede acceder directamente al repositorio del proyecto ubicado en URL: 
[https://github.com/DmyGd/ProyectoMineriaDeDatos.git](https://github.com/DmyGd/ProyectoMineriaDeDatos.git)

El proyecto consta de la siguiente estructura: 
```
MyApp.py -->EDA.py
		 -->PCA_MD.py
		 -->Kmean_md.py
		 -->ClasificaciónA.py
		 -->As_rules.py  
```
Donde el archivo **MyApp[.]py** es el archivo main donde se ejecutan todos los algoritmos.
El archivo **EDA[.]py** ejecuta el algoritmo de Análisis Exploratorio de Datos
El archivo **PCA_MD[.]py** ejecuta el algoritmo de Análisis de componentes principales.
El archivo **Kmean_md[.]py** ejecuta el algoritmo de Clustering Particional.
El archivo **ClasificaciónA[.]py** ejecuta el algoritmo de Clasificación con Regresión logística.
El archivo **As_rules[.]py** ejecuta el algoritmo de Reglas de Asociación.

### Archivo MyApp[.]py
Las librerías importadas Fueron:
```python
import streamlit as  st #Para la transformación a app web
import  pandas  as  pd  # Para la manipulacion y análisis de datos
import numpy as  np  # Para crear vectores de datos de n dimenciones
import  matplotlib.pyplot  as  plt  # Para generar gráficos
import seaborn as  sns  # Para visualización de los datos
```
Los archivos de objetos importados fueron:
```python
from  EDA  import  EDA
from  PCA_MD  import  PCA_MD
from  As_rules  import  As_rules
from  Kmean_md  import  Kmean_md
from  ClasificacionA  import  ClasificationA
```
De estos objetos importados, se obtendrán los métodos a utilizar en cada algoritmo. 

Las validaciones iniciales corresponden a la entrada de un archivo de tipo CSV, donde de ser ingresado un parámetro valido se procederá a mostrar dos opciones, la primera de ellas es "mostrar tofos los datos" y la segunda es "mostrar una cabecera", que corresponde a proyectar en la interfaz solo 5 renglones de la data ingresada
```python
file  =  st.file_uploader("Archivo", type=["csv"])
if  file  !=  None:
	data  =  pd.read_csv(file)
if  st.checkbox('Mostrar todos los datos'):
	st.write(data)
if  st.checkbox('Mostrar cabecera'):
	st.write(data.head())
```
Lo siguiente en el código corresponde a una selección box de tipo dropdown menú en la que se nos permitirá elegir el algoritmo deseado. El código correspondiente es el siguiente:

```python
# Barra de control lateral Izquierdo
# Add a selectbox to the sidebar:
add_selectbox  =  st.sidebar.selectbox(
'Algoritmo Actual', ('EDA', 'PCA', 'Reglas de asociación', 'Cluster Particional', 'Clasificación con Regresión Logística'))
st.write("""# Estas en el algoritmo: """)
st.write(f"## *{add_selectbox}*")
```
La función `st.sidebar.selectbox()` recibe dos parámetros,  el primero de ellos es el título que tendrá el dorpdown menú y el segundo es una tupla con las opciones a mostrar.

Lo siguiente en el código sera la validación y ejecución de los algoritmos según la selección del usuario:

```python
try:
	if  add_selectbox  ==  'EDA':
		eda  =  EDA(data)
		eda.algorithm_description()
		eda.description_data_structure()
		eda.missing_data()
		eda.outlier_detection()
		eda.relations_variables()
		
	if  add_selectbox  ==  'PCA':
		pca_md  =  PCA_MD(data)
		pca_md.algorithm_description()
		pca_md.data_standardization()
		pca_md.cov_components_varianza()
		pca_md.main_components()
		pca_md.relevancy_ratio_cargas_()
		pca_md.create_new_matrix()
	  
	if  add_selectbox  ==  'Reglas de asociación':
		as_rules  =  As_rules(data)
		as_rules.algorithm_description()
		as_rules.data_processing()
	  
	if  add_selectbox  ==  'Cluster Particional':
		kmean  =  Kmean_md(data)
		kmean.algorithm_description()
		kmean.feature_selection()
		kmean.kmeans_algorithm()

	if  add_selectbox  ==  'Clasificación con Regresión Logística':
		clasificacion  =  ClasificationA(data)
		clasificacion.algorithm_description()
		clasificacion.feature_selection()
		clasificacion.d_predictor_and_class_var()
		clasificacion.algorithm_application()
except:
	st.write("No se ha ingresado una fuente de información")
```

 Como se puede apreciar, la ejecución de cada uno de los algoritmos es muy similar, se crea un objeto, se le pasa la data y a partir de ese objeto, se van ejecutando los métodos asociados al algoritmo. Todo el bloque de código se ha encerrado en una excepción debido a que si el sistema no puede procesar la data, no interrumpa su ejecución y el usuario tenga una idea de que algo esta mal con su documento de datos . 

### Archivo EDA[.]py
Las librerías importadas fueron:
```python
import streamlit as  st
import  pandas  as  pd  # Para la manipulacion y análisis de datos
import numpy as  np  # Para crear vectores de datos de n dimenciones
import  matplotlib.pyplot  as  plt  # Para generar gráficos
import seaborn as  sns  # Para visualización de los datos
import altair as  alt
import  io
```
Para poder utilizar sin advertencias inesperadas por parte de la librería Streamlit, se inhabilitan con la línea:
```python
st.set_option('deprecation.showPyplotGlobalUse', False)
```
Lo siguiente en el código es un objeto que contiene todos los métodos del algoritmo, en este caso es el siguiente:

```python
class  EDA():
	def  __init__(self, data):
		self.data  =  data
```
Se puede observar que como parámetro de creación recibe la data, misma que es guardada en un atributo de la clase llamado data.

Los métodos que contiene la clase son:
```python
def  algorithm_description(self)
def  description_data_structure(self)
def  missing_data(self)
def  outlier_detection(self):
def  relations_variables(self):
```
El método `algorithm_description` ofrece una descripción del algoritmo al que pertenece.
El método `description_data_structure` ofrece una descripción de la estructura de datos
El método `missing_data` realiza una identificación de datos faltantes
El método `outlier_detection` realiza una deteccion de valores Atipicos
El método `relations_variables` realiza una identificación de relaciónes entre variables

### Archivo PCA_MD[.]py
Las librerias importadas fueron:
```python
import streamlit as  st
import  pandas  as  pd  # Para la manipulación y análisis de datos
import numpy as  np  # Para crear vectores y matrices n dimensionales
# Para la generación de gráficas a partir de los datos
import  matplotlib.pyplot  as  plt
import seaborn as  sns  # Para la visualización de datos basado en matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import  PCA
```
Para poder utilizar sin advertencias inesperadas por parte de la librería Streamlit, se inhabilitan con la línea:
```python
st.set_option('deprecation.showPyplotGlobalUse', False)
```
Siguiendo la estructura del archivo anterior, se crea una clase que como parámetro de creación recibe la data:
```python
class  PCA_MD():
	def  __init__(self, data):
		self.data  =  data
```

Los métodos contenidos en la clase son:
```python
def  matriz_nomalizada(self, Mdata)
def  get_component(self, MNormalizada, num_comp=9)
def  algorithm_description(self)
def  data_standardization(self)
def  cov_components_varianza(self)
def  main_components(self)
def  relevancy_ratio_cargas_(self)
def  create_new_matrix(self)
```

Los primeros dos métodos `matriz_nomalizada` y `get_component` corresponden a métodos internos, estos métodos obtienen una matriz normalizada y un componente repectivamente (sonutilizados internamente por el objeto).
El método `algorithm_description` ofrece una descripción del algoritmo al que pertenece.
El método `data_standardization` realiza una etandarización de los datos.
El método `cov_components_varianza`  realiza el calculo de la matriz de Covarianza, Componentes y Varianza
El método `main_components` obtiene los componentes principales.
El método `relevancy_ratio_cargas_` obtiene la proporción de relevancias –cargas–
El método `create_new_matrix` crea una nueva matriz para realizar analisis distintos con base en los resultados anteriores

### Archivo Kmean_md[.]py
Las librerías importadas fueron:
```python
import streamlit as  st
import  pandas  as  pd
import numpy as  np
import  matplotlib.pyplot  as  plt
import seaborn as  sns
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
```
Para poder utilizar sin advertencias inesperadas por parte de la librería Streamlit, se inhabilitan con la línea:
```python
st.set_option('deprecation.showPyplotGlobalUse', False)
```
Siguiendo la estructura de los archivos anteriores, se crea una clase que como parámetro de creación recibe la data:
```python
class  Kmean_md():
	def  __init__(self, data):
		self.data  =  data
		self.MatrizVariables  = []
		self.valoresSeleccionados  = []
```
A diferencia de las clases anteriores, esta crea dos atributos de tipo lista con los que traba internamente.

Los métodos contenidos en la clase son:
```python
	def algorithm_description(self)
	def feature_selection(self)
	def kmeans_algorithm(self):
```

Dada la complejidad del algoritmo, se crearon muy pocos métodos, es decir, internamente contienen todo el proceso anidado, de tal manera que no se pueden separar en mas métodos

El método `algorithm_description` ofrece una descripción del algoritmo al que pertenece.
El método `feature_selection` egloba el proceso de selección de características
El método `kmeans_algorithm` engloba tanto al algoritmo K-means como el Elbow-method


### Archivo ClasificacionA[.]py
Las librerías importadas fueron:
```python
import streamlit as  st
import  pandas  as  pd
import numpy as  np
import  matplotlib.pyplot  as  plt
import seaborn as  sns  # Para la visualización de datos basado en matplotlib
#metricas
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
```
Para poder utilizar sin advertencias inesperadas por parte de la librería Streamlit, se inhabilitan con la línea:
```python
st.set_option('deprecation.showPyplotGlobalUse', False)
```
Siguiendo la estructura de los archivos anteriores, se crea una clase que como parámetro de creación recibe la data:
```python
class  ClasificationA():
	def  __init__(self, data):
		self.data  =  data
		self.valor  =  ""
		self.X  =  ""
		self.Y  =  ""
```
Al momento de crear un objeto de tipo ClasificationA recibe como parámetro la data e internamente crea 3 atributospara trabajar los distintos métdos alojados en la clase.
 
 Los métodos contenidos en la clase son:
 
```python
def  algorithm_description(self)
def  feature_selection(self)
def  d_predictor_and_class_var(self):
def  algorithm_application(self):
```
El método `algorithm_description` ofrece una descripción del algoritmo al que pertenece.
El método `feature_selection` egloba el proceso de selección de características
El método `d_predictor_and_class_var` obtiene una definición de las variables predictoras y la variable clase
El método `algorithm_application` contiene el la aplicación del algoritmo de clasificación con regresión logística

## Stack de Funciones
**st.write(String)**, Esribe texto, figuras e imagenes en la aplicación web. 
**st.pyplot()**, Realiza el ploteo de los datos previamente preparados con matplotlib. 
**st.checkbox(String)**, Despliega un input de tipo checkbox en la aplicación web. Regresa True o False.
**st.text_input(String)**, Despliega un input text en la aplicación web, regresa la cadena ingresada
**st.markdown(String, unsafe_allow_html=True)**, Escribe texto en formato HTML en la aplicación web. 
Para mayor información relacionada con la api de Streamlit visite [https://docs.streamlit.io/en/stable/api.html](https://docs.streamlit.io/en/stable/api.html)