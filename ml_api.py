import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

def load_data(folder_path):
    csv_files = glob.glob(f'{folder_path}/*.csv')
    columns_to_keep = ['annee_numero_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
    df = pd.DataFrame()
    for file in csv_files:
        df_temp = pd.read_csv(file, encoding='ISO-8859-1', sep=';', index_col=False, usecols=columns_to_keep)
        df = pd.concat([df, df_temp], ignore_index=True)
    df.sort_values(by='annee_numero_de_tirage', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df):
    df = df.drop(['annee_numero_de_tirage'], axis=1)
    scaler = StandardScaler().fit(df.values)
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)
    return transformed_df, scaler

def prepare_training_data(transformed_df, window_length):
    number_of_rows = transformed_df.shape[0]
    number_of_features = transformed_df.shape[1]
    train = np.empty([number_of_rows-window_length, window_length, number_of_features], dtype=float)
    label = np.empty([number_of_rows-window_length, number_of_features], dtype=float)
    for i in range(0, number_of_rows-window_length):
        train[i] = transformed_df.iloc[i:i+window_length].values
        label[i] = transformed_df.iloc[i+window_length].values
    return train, label

def build_model(window_length, number_of_features):
    model = Sequential([
        LSTM(32, input_shape=(window_length, number_of_features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(number_of_features)
    ])
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def train_model(model, train, label, batch_size=64, epochs=100):
    model.fit(train, label, batch_size=batch_size, epochs=epochs)
    return model

def predict(model, scaler, df, window_length):
    to_predict = df.iloc[-window_length:]
    scaled_to_predict = scaler.transform(to_predict)
    scaled_predicted_output = model.predict(np.array([scaled_to_predict]))
    predictions = scaler.inverse_transform(scaled_predicted_output).astype(int)
    return predictions

# 以下是主流程代码，您可以根据需要调用这些函数
if __name__ == "__main__":
    folder_path = r'C:\Users\33766\Desktop\mlipproject\data'
    df = load_data(folder_path)
    transformed_df, scaler = preprocess_data(df)
    train, label = prepare_training_data(transformed_df, 5)
    model = build_model(5, transformed_df.shape[1])
    model = train_model(model, train, label)
    predictions = predict(model, scaler, transformed_df, 5)
    print(predictions)
