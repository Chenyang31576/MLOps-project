# tests/test_integration.py

import unittest
from ml_api import load_data, preprocess_data, prepare_training_data

class TestIntegration(unittest.TestCase):
    def test_data_flow(self):
        # 测试数据加载、预处理和训练数据准备的整合
        df = load_data('data')
        transformed_df, _ = preprocess_data(df)
        train, label = prepare_training_data(transformed_df, 5)
        self.assertEqual(train.shape[0], label.shape[0])  # 确保训练数据和标签的数量一致

if __name__ == '__main__':
    unittest.main()
