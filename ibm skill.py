import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pywt
import matplotlib.pyplot as plt
import seaborn as sns


def apply_cwt(data, widths):
    cwt_matrix, _ = pywt.cwt(data, widths, 'morl')
    cwt_flattened = cwt_matrix.flatten()
    return cwt_flattened


def extract_features(data, widths):
    features = []
    for recording in data:
        coeffs = apply_cwt(recording, widths)
        features.append(coeffs)
    return np.array(features)


def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    return svm_classifier, X_test, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(classification_report(y_test, y_pred, zero_division=1))

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Outage', 'Outage'],
                yticklabels=['No Outage', 'Outage'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_original_data(df):
    plt.figure(figsize=(12, 6))
    for column in df.columns[:-1]:
        plt.plot(df[column], label=column)
    plt.legend()
    plt.title('Original Data')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.show()


def plot_wavelet_transform(data, widths):
    plt.figure(figsize=(12, 6))
    cwt_matrix, _ = pywt.cwt(data, widths, 'morl')
    plt.imshow(cwt_matrix, extent=[0, len(data), 1, widths[-1]], cmap='PRGn', aspect='auto',
               vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
    plt.colorbar()
    plt.title('Continuous Wavelet Transform (CWT)')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.show()


def main():
    df = pd.read_csv('C:/Users/Debabrata/Desktop/energydata_complete.csv')
    df = df.dropna()
    df['outage'] = np.where(df['Appliances'] > df['Appliances'].quantile(0.95), 1, 0)
    X = df.drop(['outage', 'date'], axis=1).values
    y = df['outage'].values
    widths = np.arange(1, 31)

    plot_original_data(df)
    plot_wavelet_transform(X[0], widths)

    X_features = extract_features(X, widths)
    svm_model, X_test, y_test = train_svm(X_features, y)

    print("Evaluation Report:")
    evaluate_model(svm_model, X_test, y_test)

    new_data = np.array([
        [450.0, 2100.0, 1050.0, 5100.0, 49.0, 65.0],
        [430.0, 2200.0, 1100.0, 5200.0, 39.0, 55.0]
    ])
    new_data_features = extract_features(new_data, widths)
    predicted_faults = svm_model.predict(new_data_features)
    print("Predicted faults for new data:", predicted_faults)


if __name__ == "__main__":
    main()
