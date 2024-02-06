import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout

# Constants
FOLDER_PATH = r'C:\Users\33766\Desktop\mlipproject\data'
CSV_FILES = glob.glob(f'{FOLDER_PATH}/*.csv')
COLUMNS_TO_KEEP = ['annee_numero_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']

def load_and_prepare_data(csv_files, columns_to_keep):
    df = pd.DataFrame()
    for file in csv_files:
        df_temp = pd.read_csv(file, encoding='ISO-8859-1', sep=';', index_col=False, usecols=columns_to_keep)
        df = pd.concat([df, df_temp], ignore_index=True)
    df.sort_values(by='annee_numero_de_tirage', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def preprocess_data(df):
    # 假设 'annee_numero_de_tirage' 是年份+编号的格式，先提取年份
    df['year'] = df['annee_numero_de_tirage'].apply(lambda x: str(x)[:4])
    df = df.drop(['annee_numero_de_tirage', 'year'], axis=1)
    scaler = StandardScaler().fit(df.values)
    transformed_dataset = scaler.transform(df.values)
    transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)
    return transformed_df, scaler

def prepare_training_data(transformed_df, window_length, number_of_features):
    number_of_rows = transformed_df.shape[0]
    train = np.empty([number_of_rows-window_length, window_length, number_of_features], dtype=float)
    label = np.empty([number_of_rows-window_length, number_of_features], dtype=float)
    for i in range(0, number_of_rows-window_length):
        train[i] = transformed_df.iloc[i:i+window_length].values
        label[i] = transformed_df.iloc[i+window_length].values
    return train, label

def create_and_train_model(train, label, number_of_features, window_length, model_file):
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        model = Sequential([
            LSTM(32, input_shape=(window_length, number_of_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(number_of_features)
        ])
        model.compile(loss='mse', optimizer='rmsprop')
        model.fit(train, label, batch_size=64, epochs=100)
        model.save(model_file)
    return model

def predict(model, scaler, df, window_length):
    to_predict = df.iloc[-window_length:]
    scaled_to_predict = scaler.transform(to_predict)
    scaled_predicted_output = model.predict(np.array([scaled_to_predict]))
    predicted_data = scaler.inverse_transform(scaled_predicted_output).astype(int)
    predict = pd.DataFrame(predicted_data, columns=['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2'])
    return predict

# Main execution
if __name__ == "__main__":
    df = load_and_prepare_data(CSV_FILES, COLUMNS_TO_KEEP)
    transformed_df, scaler = preprocess_data(df)
    train, label = prepare_training_data(transformed_df, window_length=5, number_of_features=len(COLUMNS_TO_KEEP)-2)
    model = create_and_train_model(train, label, number_of_features=len(COLUMNS_TO_KEEP)-2, window_length=5, model_file='euromillions.h5')
    predictions = predict(model, scaler, df, window_length=5)
    print(predictions)
