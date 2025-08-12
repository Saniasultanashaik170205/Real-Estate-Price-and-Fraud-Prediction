import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

data = pd.read_csv(r"fraud.csv")

data = data.dropna(subset=['Fraud_Probability_Score'])
target = 'Fraud_Probability_Score'
categorical_features = [
    'Owner_Verified', 'Legal_Issues', 'Property_Status',
    'Amenities_Available', 'Government_Approval_Status', 'Disputed_Ownership_Claims', 'Property_Tax_Paid_Status', 'Mortgage_Status'
]
numerical_features = [
    'Price', 'Size_in_Sqft', 'Listing_Age_Days', 'Building_Age',
    'Number_of_Complaints_Filed', 'Agent_Rating'
]
selected_features = categorical_features + numerical_features
data = data.dropna(subset=selected_features)
X = data[selected_features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1))
])
pipeline.fit(X_train, y_train)
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump((pipeline, selected_features), f)
print("Model trained and saved as fraud_model.pkl")
