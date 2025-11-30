import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("Dataset/WineQT.csv")
x = df.drop(["quality", "Id"], axis=1)
y = df["quality"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# Melatih model naive bayes
model = GaussianNB()
model.fit(x_train, y_train)
joblib.dump(model, "NaiveBayesWine.pkl")


y_pred = model.predict(x_test)

# Evaluasi
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report")
classification_report(y_test, y_pred)
