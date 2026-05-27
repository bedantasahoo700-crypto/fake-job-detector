import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("fake_job_postings.csv")

df = df[['title','location','department','salary_range',
         'company_profile','description','requirements',
         'benefits','employment_type','required_experience',
         'required_education','industry','function','fraudulent']]

df = df.fillna("Unknown")

X = df.drop("fraudulent", axis=1)
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

cat_features = X.select_dtypes(include=['object']).columns
cat_idx = [X.columns.get_loc(c) for c in cat_features]

model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, verbose=100)

model.fit(X_train, y_train, cat_features=cat_idx)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved!")