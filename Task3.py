# =====================================
# 1. Import Libraries
# =====================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =====================================
# 2. Load Data
# =====================================
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url)

print("First 5 Rows:\n", df.head())

# =====================================
# 3. Explore Data
# =====================================
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistics:\n", df.describe())

# Scatter plots (to see relationships)
sns.pairplot(df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']])
plt.show()

# =====================================
# 4. Handle Missing Values
# =====================================
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

# =====================================
# 5. Prepare Data
# =====================================
X = df[['displacement', 'horsepower', 'weight', 'acceleration']]
y = df['mpg']

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# 6. Polynomial Feature Transformation
# =====================================
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# =====================================
# 7. Build Model
# =====================================
model = LinearRegression()
model.fit(X_train_poly, y_train)

# =====================================
# 8. Predict
# =====================================
y_pred = model.predict(X_test_poly)

# =====================================
# 9. Evaluate Model
# =====================================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# =====================================
# 10. Interpret (Feature Effect)
# =====================================
feature_names = poly.get_feature_names_out(X.columns)

coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})

print("\nFeature Impact:\n", coeff_df.head(10))

# =====================================
# 11. Actual vs Predicted Plot
# =====================================
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG (Polynomial Regression)")
plt.show()
