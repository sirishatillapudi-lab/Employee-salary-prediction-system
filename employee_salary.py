import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("salary.csv")
X = df[["experience_years", "education_level", "job_role"]]
y = df["salary"]
prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"),
                           ["education_level", "job_role"])], remainder="passthrough")
model = make_pipeline(prep, RandomForestRegressor(random_state=42))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(Xtr, ytr)
print("MAE:", round(mean_absolute_error(yte, model.predict(Xte)), 2))
sample = pd.DataFrame([{"experience_years": float(input("Experience years: ")),
 "education_level": input("Education (BTech/MTech/Degree): "),
 "job_role": input("Role (Developer/Tester/Analyst): ")}])
print("Predicted salary:", round(model.predict(sample)[0], 2))
