import pickle
import pandas as pd
import streamlit as st

# Cargar el modelo preentrenado y el scaler
with open('model_lgbm_op_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_op.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Título de la aplicación
st.title('Predicción de Impagos en Préstamos Personales')

# Estilos adicionales para mejorar la vista
st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        color: #FF5733;
    }
    .prob-font {
        font-size:30px !important;
        color: #2ECC71;
    }
    </style>
    """, unsafe_allow_html=True)

# Entrada de datos del usuario numéricos con valores por defecto
age = st.number_input('Edad', min_value=0)
income = st.number_input('Ingreso Anual', min_value=0.0)
loan_amount = st.number_input('Monto del Préstamo', min_value=0.0)
credit_score = st.number_input('Puntuación de Crédito', min_value=0, value=500)  # Valor por defecto 500
months_employed = st.number_input('Meses Empleados', min_value=0)
num_credit_lines = st.number_input('Número de Líneas de Crédito', min_value=0)
interest_rate = st.number_input('Tasa de Interés', min_value=0.0, value=10.0)  # Valor por defecto 10
loan_term = st.number_input('Duración del Préstamo (meses)', min_value=0)
dti_ratio = st.number_input('Relación Deuda-Ingreso (DTI)', min_value=0.0, value=0.5)  # Valor por defecto 0.5

# Entrada de datos del usuario categóricos
education = st.selectbox('Nivel Educativo', ["Bachillerato", "Universidad", "Master", "Doctorado"])
employment_type = st.selectbox('Tipo de Empleo', ["Medio Tiempo", "Desempleado", "Autónomo", "Tiempo Completo"])
marital_status = st.selectbox('Estado Civil', ["Casado", "Divorciado", "Soltero"])
has_mortgage = st.selectbox('Tiene Hipoteca', ["Sí", "No"])
has_dependents = st.selectbox('Tiene Dependientes', ["Sí", "No"])
loan_purpose = st.selectbox('Finalidad del Préstamo', ["Negocios", "Compra de Vivienda", "Educación", "Compra Automóvil", "Otros"])
has_co_signer = st.selectbox('Tiene Avalista', ["Sí", "No"])

# Selector para Cluster con valores transformados
cluster = st.selectbox('Cluster', [1, 2, 3])  # Solo puede elegir entre 1, 2 o 3
cluster_transformed = cluster - 1  # Transformar el cluster: 1 -> 0, 2 -> 1, 3 -> 2

# Diccionario de mapeo para variables categóricas
education_mapping = {'Bachillerato': 1, 'Universidad': 2, 'Master': 3, 'Doctorado': 4}
employment_mapping = {'Medio Tiempo': 1, 'Desempleado': 2, 'Autónomo': 3, 'Tiempo Completo': 4}
marital_status_mapping = {'Casado': 1, 'Divorciado': 2, 'Soltero': 3}
has_mortgage_mapping = {'Sí': 1, 'No': 0}
has_dependents_mapping = {'Sí': 1, 'No': 0}
loan_purpose_mapping = {'Negocios': 1, 'Compra de Vivienda': 2, 'Educación': 3, 'Compra Automóvil': 4, 'Otros': 5}
has_co_signer_mapping = {'Sí': 1, 'No': 0}

# Convertir las entradas categóricas en valores numéricos
education_num = education_mapping[education]
employment_type_num = employment_mapping[employment_type]
marital_status_num = marital_status_mapping[marital_status]
has_mortgage_num = has_mortgage_mapping[has_mortgage]
has_dependents_num = has_dependents_mapping[has_dependents]
loan_purpose_num = loan_purpose_mapping[loan_purpose]
has_co_signer_num = has_co_signer_mapping[has_co_signer]

# Crear nuevas variables derivadas de las originales
loan_amount_to_income = loan_amount / income if income > 0 else 0
interest_rate_term = interest_rate * loan_term if loan_term > 0 else 0
dti_ratio_to_loan_amount = dti_ratio / loan_amount if loan_amount > 0 else 0
credit_score_income = credit_score * income if income > 0 else 0
age_employment_type = age * employment_type_num

# Crear un DataFrame con las variables finales
data = {
    'Age': [age],
    'Income': [income],
    'LoanAmount': [loan_amount],
    'CreditScore': [credit_score],
    'MonthsEmployed': [months_employed],
    'NumCreditLines': [num_credit_lines],
    'InterestRate': [interest_rate],
    'LoanTerm': [loan_term],
    'DTIRatio': [dti_ratio],
    'Education': [education_num],
    'EmploymentType': [employment_type_num],
    'MaritalStatus': [marital_status_num],
    'HasMortgage': [has_mortgage_num],
    'HasDependents': [has_dependents_num],
    'LoanPurpose': [loan_purpose_num],
    'HasCoSigner': [has_co_signer_num],
    'Cluster': [cluster_transformed],  # Usamos el valor transformado
    'LoanAmount_to_Income': [loan_amount_to_income],
    'InterestRate_Term': [interest_rate_term],
    'DTIRatio_to_LoanAmount': [dti_ratio_to_loan_amount],
    'CreditScore_Income': [credit_score_income],
    'Age_EmploymentType': [age_employment_type],
}

df = pd.DataFrame(data)

# Aplicar la estandarización usando el scaler cargado
df_scaled = scaler.transform(df)

# Hacer la predicción con los datos escalados
prediction = model.predict(df_scaled)
prediction_proba = model.predict_proba(df_scaled)

# Mostrar los resultados con estilo mejorado
st.markdown(f"<div class='big-font'>Predicción: {'Impago' if prediction[0] == 1 else 'No Impago'}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='prob-font'>Probabilidad de Impago: {prediction_proba[0][1]:.2%}</div>", unsafe_allow_html=True)
