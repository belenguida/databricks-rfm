# Databricks notebook source
# MAGIC %sh git init
# MAGIC git add *
# MAGIC git commit -m “first commit”
# MAGIC git remote add origin https://github.com/belenguida/databricks-rfm
# MAGIC git push -u origin master

# COMMAND ----------

# En esta notebook se genera la tabla final con los segmentos de clientes según el modelo RFM
# Acciones que se ejecutan en esta notebook:
# 1. Inicializar la instancia de Azure Machine Learning Service
# 2. A partir de los datos extraidos y preprocesados anteriormente (ver pipeline) se ejecuta el modelo RFM
# 3. Deploy del modelo 
# 

# actualizar key vault con las claves
workspace="RFM_test"
subscription_id= "275232e0-797c-4edb-9792-7e34f3eb5434"#"7a9aeede-b4a3-4258-9edd-d15bc7318a64"
resource_grp="rfm_test"

experiment_name = "rfm_model_int"
model_name = "rfmmodel.mml" # in case you want to change the name, keep the .mml extension

import azureml.core
from azureml.core import Workspace
from azureml.core.run import Run
from azureml.core.experiment import Experiment

# Check core SDK version number
#print("SDK version:", azureml.core.VERSION)

import azureml.core
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies

from azureml.core.authentication import InteractiveLoginAuthentication
ia = InteractiveLoginAuthentication(tenant_id='275232e0-797c-4edb-9792-7e34f3eb5434')

#ws = Workspace.create(name='rfm-ws',
  #                    subscription_id='7a9aeede-b4a3-4258-9edd-d15bc7318a64', 
   #                   resource_group='rfm-ws',
    #                  create_resource_group=True,
     #                 location='northeurope',
      #                auth=ia              )

ws = Workspace.get(name='RFM_test',
                      subscription_id='7a9aeede-b4a3-4258-9edd-d15bc7318a64', 
                      resource_group='rfm_test',
                      auth=ia
                     )

ws.get_details()

print("Found workspace {} at location {}".format(ws.name, ws.location))

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# COMMAND ----------

df_trans = spark.sql("SELECT idtransaccion, importetc, anio, lpad(ltrim(mes), 2, 0) as mes, lpad(ltrim(dia), 2, 0) as dia FROM resumen2019 where siteid <> 'siteid'")
df_tienda = spark.sql("SELECT * FROM customer_transaccion_tienda")

transacciones_tienda = df_tienda.toPandas()
trans_importe = df_trans.toPandas()

# COMMAND ----------

dbutils.library.installPyPI("mlflow")

import mlflow
import mlflow.sklearn
# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
mlflow.start_run()

# COMMAND ----------




# Importamos las librerias necesarias para ejecutar el modelo 
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans, AgglomerativeClustering
#import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
warnings.filterwarnings('ignore')

transacciones_tienda = df_tienda.toPandas()
trans_importe = df_trans.toPandas()

#transacciones_tienda = pd.read_csv('/dbfs/FileStore/tables/df_customer_transaccion_tienda.csv')
transacciones_tienda = transacciones_tienda.drop_duplicates().reset_index(drop=True)

#trans_importe = pd.read_csv('/dbfs/FileStore/tables/resumen2019__1_.csv',usecols =['idtransaccion','importetc', 'anio', 'mes', 'dia'])
                       

trans_importe = trans_importe.drop_duplicates().reset_index(drop=True)
trans_importe=trans_importe.rename(columns = {'idtransaccion':'id_transaccion', 'importetc':'total'})

trans_importe['fecha_transaccion'] = pd.to_datetime(trans_importe['anio'].astype(str) + trans_importe.mes.astype(str) + trans_importe.dia.astype(str), format="%Y/%m/%d")
trans_importe = trans_importe.drop(columns=['anio', 'mes', 'dia'])

#trans_importe
#trans_importe = trans_importe[trans_importe['total'] > 0.01]

transacciones_tienda = pd.merge(transacciones_tienda, trans_importe, on = 'id_transaccion')
transacciones_tienda

transacciones_tienda['total'] = transacciones_tienda['total'].apply(pd.to_numeric, errors='coerce')

#Se sumarizan los totales por customerID
rfm = transacciones_tienda.groupby('customer_id').total.agg(['sum','count']).reset_index()
rfm = rfm.rename(columns = {'sum':'monetary', 'count':'frequency'})

max_transaccion = trans_importe['fecha_transaccion'].max()
rfm['recency'] = (max_transaccion - trans_importe['fecha_transaccion']).dt.days

rfm_cols = rfm.filter(['monetary','frequency','recency'])
#rfm_cols = rfm_cols.apply(pd.to_numeric, errors='coerce')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
rfm_norm = pd.DataFrame(scaler.fit_transform(rfm_cols))
rfm_norm.columns = ['monetary_n','frequency_n','recency_n']

#rfm_norm.fillna(1, inplace=True)
rfm_norm.dropna(inplace=True)

rfm_norm = rfm_norm.reset_index(drop=True)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sse = []
models = []

range_n_clusters = [4,5]

for k in range_n_clusters:

    kmeans = KMeans(n_clusters = k, max_iter = 30)
    klabels = kmeans.fit_predict(rfm_cols)
    sse.append(kmeans.inertia_)
    models.append(kmeans)
    
    silhouette_avg = silhouette_score(pd.DataFrame(rfm_cols), klabels)
    print("For k =", k, "The average silhouette_score is :", silhouette_avg)
    
model = (KMeans(n_clusters = 4, max_iter = 30)).fit(rfm_norm)
centers = model.cluster_centers_
fig = plt.figure()
ax = fig.add_subplot(111,alpha=0.2)
ax.scatter(rfm_norm['monetary_n'],rfm_norm['frequency_n'],rfm_norm['recency_n'], cmap='viridis', c = model.predict(rfm_norm))
#ax.scatter(rfm_cols['monetary'],rfm_cols['frequency'],rfm_cols['recency'], cmap='brg', c = model.predict(rfm_cols))
ax.scatter(centers[:,0],centers[:,1],c='black')

rfm_cluster = pd.DataFrame(scaler.inverse_transform(rfm_norm))
rfm_cluster.columns = rfm_cols.columns
rfm_cluster['cluster'] = model.labels_
rfm_cluster['customer_id'] = rfm['customer_id']

rfm_cluster =  rfm_cluster[['customer_id', 'cluster', 'monetary','frequency','recency']]



# COMMAND ----------

rfm_resumen = rfm_cluster.groupby('cluster').agg({
    'frequency':['mean', 'min', 'max'],
    'recency':['mean', 'min', 'max'],
    'monetary':['mean', 'min', 'max', 'count']
})

rfm_resumen.to_csv('/dbfs/FileStore/rfm_resumen.csv')

mlflow.log_artifact('/dbfs/FileStore/rfm_resumen.csv')

# COMMAND ----------

rfm_resumen

# COMMAND ----------

#Registro del modelo - aca se puede parametrizar el nombre y despues usarlo para filtrar y asi tener varios modelos
# al mismo tiempo
run_id = mlflow.search_runs().iloc[0].run_id

model_name = "rfm-test"
model_version = mlflow.register_model(f"runs:/{run_id}", model_name)

mlflow.end_run()

# COMMAND ----------

#Stage del modelo en producción
from mlflow.tracking import MlflowClient
 
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=6,
  stage="Production",
)


# COMMAND ----------

#Referencia al modelo ya registrado en prod
model = mlflow.pyfunc.load_model(f"models:/{model_name}/7")
 
# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {model}')

# COMMAND ----------

# prueba llamada al servicio

import os
import requests
import pandas as pd

def score_model():
  url = 'https://adb-3700965641645369.9.azuredatabricks.net/2.0/mlflow/artifacts/list'
  #https://adb-3700965641645369.9.azuredatabricks.net/model/rfm-test/Production/invocations
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  response = requests.request(method='GET', headers=headers, 
                              url=url)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response#.json()

resultados = score_model()
resultados.content


# COMMAND ----------

