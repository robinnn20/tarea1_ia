import bnlearn as bn
import pandas as pd

# Cargar el dataset de Adult
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
    'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
    'hours_per_week', 'native_country', 'income'
]
df_adult = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

# Eliminar filas con valores nulos
df_adult.dropna(inplace=True)

# Convertir columnas a tipo categórico
categorical_cols = ['workclass', 'education', 'marital_status', 'occupation',
                    'relationship', 'race', 'sex', 'native_country', 'income']
df_adult[categorical_cols] = df_adult[categorical_cols].astype('category')
# Seleccionar un subconjunto de columnas de interés
df_subset = df_adult[['workclass', 'education', 'occupation', 'income','sex']]
# 3.1 ExhaustiveSearch (nombre correcto es 'ex')
model_exhaustive = bn.structure_learning.fit(df_subset, methodtype='ex')

# 3.2 HillClimbSearch (nombre correcto es 'hc')
model_hillclimb = bn.structure_learning.fit(df_subset, methodtype='hc')

# Aprender los parámetros para ambas redes
model_exhaustive_params = bn.parameter_learning.fit(model_exhaustive, df_subset)
model_hillclimb_params = bn.parameter_learning.fit(model_hillclimb, df_subset)

# Inferencia: Definir las condiciones para las inferencias
# Inferencia 1: ¿Cuál es la probabilidad de que una persona gane más de 50K dado que es de educación universitaria?
query_1_exhaustive = bn.inference.fit(model_exhaustive_params, variables=['income'], evidence={'education': 'Bachelors'})
query_1_hillclimb = bn.inference.fit(model_hillclimb_params, variables=['income'], evidence={'education': 'Bachelors'})

# Inferencia 2: ¿Cuál es la probabilidad de que una persona gane más de 50K dado que es mujer?
query_2_exhaustive = bn.inference.fit(model_exhaustive_params, variables=['income'], evidence={'sex': 'Female'})
query_2_hillclimb = bn.inference.fit(model_hillclimb_params, variables=['income'], evidence={'sex': 'Female'})

# Mostrar resultados
print("Inferencia 1 (ExhaustiveSearch):", query_1_exhaustive)
print("Inferencia 1 (HillClimbSearch):", query_1_hillclimb)
print("Inferencia 2 (ExhaustiveSearch):", query_2_exhaustive)
print("Inferencia 2 (HillClimbSearch):", query_2_hillclimb)
