import pandas as pd

# Read dataset with tab separator
data = pd.read_csv("data.csv", delimiter="\t")

# Show first 5 rows
print(data.head())

# Show column names
print(data.columns)

# Select columns
X = data[["reading score", "writing score"]]

# Target column
y = data["math score"]

# Import model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Output predictions
print("Predicted Math Scores:")
print(predictions)
from sklearn.metrics import r2_score
score=r2_score(y_test,predictions)
print("Accuracy Score:",score)
reading=float(input("Enter Reading Score:"))
writing=float(input("Enter writing Score:"))
result=model.predict([[reading,writing]])
print("Predicted Math Score:",result[0])
import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.title("Student Performance Prediction")

# Save graph
plt.savefig("graph.png")

# Save predictions
output = pd.DataFrame({
    "Actual Math Score": y_test.values,
    "Predicted Math Score": predictions
})

output.to_csv("predictions.csv", index=False)

print("Files saved successfully!")
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,predictions)
print("Accuracy:",accuracy)
import joblib
joblib.dump(model,"student_model.pkl")




