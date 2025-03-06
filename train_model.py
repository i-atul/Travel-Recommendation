import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load your data
df = pd.read_csv("Notebook/merge.csv")

# Define your features and target
features = ['Name_x', 'State', 'Type', 'BestTimeToVisit', 'Preferences', 'Gender', 'NumberOfAdults', 'NumberOfChildren']
X = df[features]
y = df['Popularity']

# Encode categorical features
label_encoders = {}
for feature in features:
    if X[feature].dtype == 'object':
        le = LabelEncoder()
        X.loc[:, feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Save the model
with open('Notebook/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the label encoders
with open('Notebook/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)