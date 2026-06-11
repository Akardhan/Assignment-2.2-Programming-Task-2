import unittest
import numpy as np

from numcompute_stream.preprocessing import StandardScaler, SimpleImputer, OneHotEncoder


class TestPreprocessing(unittest.TestCase):

    def test_standard_scaler_fit_transform_shape(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        self.assertEqual(Xt.shape, X.shape)

    def test_standard_scaler_mean_close_zero(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        self.assertTrue(np.allclose(np.mean(Xt, axis=0), 0.0))

    def test_standard_scaler_partial_fit(self):
        X1 = np.array([[1, 2], [3, 4]], dtype=float)
        X2 = np.array([[5, 6], [7, 8]], dtype=float)
        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)
        self.assertEqual(scaler.n_features_in_, 2)
        self.assertTrue(np.all(scaler.var_ >= 0))

    def test_standard_scaler_handles_nan(self):
        X = np.array([[1, np.nan], [3, 4], [5, 6]], dtype=float)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        self.assertFalse(np.isnan(Xt).any())

    def test_standard_scaler_zero_variance(self):
        X = np.array([[2, 2], [2, 2], [2, 2]], dtype=float)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X)
        self.assertTrue(np.allclose(Xt, 0.0))

    def test_simple_imputer_mean(self):
        X = np.array([[1, np.nan], [3, 4], [5, 6]], dtype=float)
        imputer = SimpleImputer(strategy="mean")
        Xt = imputer.fit_transform(X)
        self.assertFalse(np.isnan(Xt).any())
        self.assertAlmostEqual(Xt[0, 1], 5.0)

    def test_simple_imputer_constant(self):
        X = np.array([[1, np.nan], [3, 4]], dtype=float)
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
        Xt = imputer.fit_transform(X)
        self.assertEqual(Xt[0, 1], -1)

    def test_simple_imputer_partial_fit(self):
        X1 = np.array([[1, np.nan], [3, 4]], dtype=float)
        X2 = np.array([[5, 6]], dtype=float)
        imputer = SimpleImputer(strategy="mean")
        imputer.partial_fit(X1)
        imputer.partial_fit(X2)
        Xt = imputer.transform(np.array([[np.nan, np.nan]]))
        self.assertFalse(np.isnan(Xt).any())

    def test_one_hot_encoder(self):
        X = np.array([["red"], ["blue"], ["red"]], dtype=object)
        encoder = OneHotEncoder()
        Xt = encoder.fit_transform(X)
        self.assertEqual(Xt.shape[0], 3)
        self.assertEqual(Xt.shape[1], 2)

    def test_one_hot_encoder_partial_fit_new_category(self):
        X1 = np.array([["red"], ["blue"]], dtype=object)
        X2 = np.array([["green"]], dtype=object)
        encoder = OneHotEncoder()
        encoder.partial_fit(X1)
        encoder.partial_fit(X2)
        Xt = encoder.transform(np.array([["green"]], dtype=object))
        self.assertEqual(Xt.shape[1], 3)
        self.assertEqual(np.sum(Xt), 1.0)


if __name__ == "__main__":
    unittest.main()