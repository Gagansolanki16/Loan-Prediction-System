import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle as pk

# Load Data
data = pd.read_csv('updated_loan_approval_dataset.csv')

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

#print("Available columns in dataset:", data.columns)

# Drop unnecessary columns and create 'Assets'
data.drop(columns=['loan_id'], inplace=True)
data['Assets'] = (data['residential_assets_value'] + 
                  data['commercial_assets_value'] + 
                  data['luxury_assets_value'] + 
                  data['bank_asset_value'])

data.drop(columns=['residential_assets_value', 'commercial_assets_value', 
                   'luxury_assets_value', 'bank_asset_value'], inplace=True)

# Clean and Encode Categorical Columns
data['education'] = data['education'].str.strip().replace({'Graduate': 1, 'Not Graduate': 0})
data['self_employed'] = data['self_employed'].str.strip().replace({'Yes': 1, 'No': 0})
data['loan_status'] = data['loan_status'].str.strip().replace({'Approved': 1, 'Rejected': 0})

# Ensure numerical columns are of the correct type
numerical_columns = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets']
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric)

# Split Data
X = data.drop(columns=['loan_status'])
y = data['loan_status']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)
print("Model Accuracy:", model.score(x_test_scaled, y_test))

# Predict on New Data
new_data = pd.DataFrame([[2, 1, 0, 9600000, 29900000, 12, 778, 50700000]], 
                        columns=['no_of_dependents', 'education', 'self_employed', 
                                 'income_annum', 'loan_amount', 'loan_term', 
                                 'cibil_score', 'Assets'])

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Loan Prediction:", "Approved" if prediction[0] == 1 else "Rejected")

# Save Model and Scaler
pk.dump(model, open('model.pkl', 'wb'))
pk.dump(scaler, open('scaler.pkl', 'wb'))
