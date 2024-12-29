import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the CSV file
data = pd.read_csv('exercise_angles.csv')  # Replace with your file path

# Inspect the first few rows


# Initialize the label encoder
label_encoder = LabelEncoder()
data['Encoded_Side'] = label_encoder.fit_transform(data['Side'])
# Transform the target labels
data['Encoded_Label'] = label_encoder.fit_transform(data['Label'])
print(dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Check the mapping
# print(dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))
# print(data.head())`

# Features (angles) and target variable
X = data.iloc[:, 1:-3].values  # Exclude non-numerical columns and target label
y = data['Encoded_Label'].values

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data, transform the test data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

# Train the SVM
svm_model.fit(X_train, y_train)

# Evaluate on test data
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler saved successfully")

# shoulder_angle, elbow_angle, hip_angle, knee_angle, ankle_angle,
#      shoulder_ground_angle, elbow_ground_angle, hip_ground_angle,
#       knee_ground_angle, ankle_ground_angle
