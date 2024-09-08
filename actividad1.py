
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
df_subset = df_adult[['age', 'workclass', 'education', 'occupation', 'income']]

# 3.1 ExhaustiveSearch (nombre correcto es 'ex')
model_exhaustive = bn.structure_learning.fit(df_subset, methodtype='ex')

# 3.2 HillClimbSearch (nombre correcto es 'hc')
model_hillclimb = bn.structure_learning.fit(df_subset, methodtype='hc')

# Mostrar estructuras aprendidas
bn.plot(model_exhaustive)
bn.plot(model_hillclimb)
