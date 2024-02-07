# tests/test_model.py

import unittest
from ml_api import load_data, preprocess_data, build_model
import os

class TestLoadData(unittest.TestCase):
    def test_load_data(self):
        # 确保load_data函数能正确加载数据
        folder_path = 'data'  # 指向包含测试CSV文件的目录
        df = load_data(folder_path)
        self.assertFalse(df.empty)  # 确保DataFrame不为空

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        # 确保preprocess_data函数能正确预处理数据
        df = load_data('data')
        transformed_df, _ = preprocess_data(df)
        self.assertEqual(transformed_df.shape[1], 7)  # 假设预期的特征数量是7

class TestBuildModel(unittest.TestCase):
    def test_build_model(self):
        # 确保build_model函数能正确构建模型
        model = build_model(5, 7)
        self.assertIsNotNone(model)  # 确保模型非空

if __name__ == '__main__':
    unittest.main()
