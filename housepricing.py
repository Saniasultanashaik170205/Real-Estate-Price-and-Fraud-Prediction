import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle

data = pd.read_csv(r"House Price India.csv")
data['total_area'] = data['living area'] + data['Area of the basement']
data['property_age'] = 2023 - data['Built Year']
data['renovated'] = (data['Renovation Year'] > 0).astype(int)

features = [
    'number of bedrooms', 'number of bathrooms', 'living area',
    'lot area', 'waterfront present', 'grade of the house',
    'total_area', 'property_age', 'renovated',
    'Number of schools nearby', 'Distance from the airport'
]
X = data[features]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ))
])
pipeline.fit(X_train, y_train)
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump((pipeline, features), f)  # Save both model and feature list
print("Model built with schools and airport distance features")
