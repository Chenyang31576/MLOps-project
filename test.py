import unittest
import os
from jkl import load_and_prepare_data, preprocess_data, prepare_training_data, create_and_train_model, predict
import pandas as pd
import numpy as np

COLUMNS_TO_KEEP = ['annee_numero_de_tirage', 'boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5', 'etoile_1', 'etoile_2']
class TestEuroMillions(unittest.TestCase):

    def test_load_and_prepare_data(self):
        # 测试数据加载和准备
        test_csv_files = ['path_to_test_csv.csv']  # Use a small test CSV file
        df = load_and_prepare_data(test_csv_files, COLUMNS_TO_KEEP)
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)

    def test_preprocess_data(self):
        # 测试数据预处理功能
        # 使用少量的测试数据来进行测试
        test_data = pd.DataFrame({
            'annee_numero_de_tirage': ['200101', '200102'],
            'boule_1': [1, 2],
            'boule_2': [2, 3],
            'boule_3': [1, 2],
            'boule_4': [2, 3],
            'boule_5': [1, 2],
            'etoile_1': [11, 9],
            'etoile_2': [3, 8]
        })
        transformed_df, scaler = preprocess_data(test_data)
        self.assertEqual(transformed_df.shape, test_data.drop(['annee_numero_de_tirage'], axis=1).shape)

    # Add more tests for other functions like prepare_training_data, create_and_train_model, predict

if __name__ == '__main__':
    unittest.main()
