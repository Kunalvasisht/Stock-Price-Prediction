import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import sys

def preprocess_data(data):
    # Preprocessing steps
    data = data.drop(['Symbol', 'Series'], axis=1)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['Weekday'] = data['Date'].dt.weekday  # Monday=0, Sunday=6
    data = data.drop(['Date'], axis=1)
    data = data.dropna()
    return data

def train_model(X, y):
    # Define preprocessing pipeline
    numeric_features = ['Year', 'Month', 'Day', 'Weekday', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])

    # Define SVR model pipeline
    svm_model = Pipeline(steps=[('preprocessor', preprocessor),('regressor', SVR(kernel='rbf'))])

    # Fit the model
    svm_model.fit(X, y)
    return svm_model

def plot_actual_vs_predicted(data, X_test, y_pred):
    # Plot actual vs predicted closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Actual Close', color='blue')
    plt.plot(X_test.index, y_pred, label='Predicted Close', color='red')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot to a file
    plot_path = "prediction_plot.png"
    plt.savefig(plot_path)
    
    # Print path to the plot image
    print(plot_path)

def main(file_path):
    # Read the data
    data = pd.read_csv(file_path)

    # Preprocess data
    data = preprocess_data(data)

    # Define features and target
    features = ['Year', 'Month', 'Day', 'Weekday', 'Prev Close', 'Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble']
    X = data[features]
    y = data['Close']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    svm_model = train_model(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")

    # Plot actual vs predicted closing prices
    plot_actual_vs_predicted(data, X_test, y_pred)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python backend.py <file_path>")
        sys.exit(1)
    file_path = sys.argv[1]
    main(file_path)