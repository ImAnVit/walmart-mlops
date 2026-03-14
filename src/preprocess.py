import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data():

    df = pd.read_csv("data/walmart.csv")

    # Adjust to the correct date format (day-month-year)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)

    df = df.drop(columns=["Date"])

    X = df.drop("Weekly_Sales", axis=1)
    y = df["Weekly_Sales"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    joblib.dump((X_train, X_test, y_train, y_test), "data/train_test.pkl")

if __name__ == "__main__":
    preprocess_data()