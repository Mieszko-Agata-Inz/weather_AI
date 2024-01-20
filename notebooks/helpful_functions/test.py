import unittest
import pandas as pd
import numpy as np
from helpful_functions import transform_data, min_max_normalize, min_max_denormalization

#######################################################################################
################################ Script with tests ####################################
#######################################################################################


# min_max_normalize
class TestMinMaxNormalize(unittest.TestCase):
    def test_list_int(self):
        """
        Test that the function transforms data properly
        """
        # min == [0, 0] and max == [2, 2]
        max = [2, 2]
        min = [0, 0]

        X_input = np.array([[[0, 0], [2, 2], [2, 2]], [[2, 2], [2, 2], [4, 4]]])
        y_input = np.array([[4, 4], [6, 6]])
        index1 = 0
        index2 = 1

        # output
        X = np.array([[[0, 0], [1, 1], [1, 1]], [[1, 1], [1, 1], [2, 2]]])
        y = np.array([[2, 2], [3, 3]])

        result_X, result_y = min_max_normalize(
            X_input, y_input, index1, max[index1], min[index2]
        )
        result_X, result_y = min_max_normalize(
            result_X, result_y, index2, max[index1], min[index2]
        )

        self.assertTrue((result_X == X).all())
        self.assertTrue((result_y == y).all())


# min_max_denormalization
class TestMinMaxDenormalization(unittest.TestCase):
    def test_list_int(self):
        """
        Test that the function transforms data properly
        """
        # min == [0, 0] and max == [2, 2]
        max = np.array([2, 2])
        min = np.array([0, 0])
        y_input = np.array([[2, 2], [3, 3]])

        # output
        y = np.array([[4, 4], [6, 6]])

        result_y = min_max_denormalization(y_input, max, min)

        self.assertTrue((result_y == y).all())


# transform_data
class TestTransformData(unittest.TestCase):
    def test_list_int(self):
        """
        Test that the function transforms dataframe properly
        and returns data splitted with connection to window size
        """

        df = pd.DataFrame()
        df["col1"] = range(10)
        df["col2"] = range(10)
        window_size = 3

        # min == [0, 0] and max == [1, 1] so as to function
        # min_max_normalize has no influence on data
        max = [1, 1]
        min = [0, 0]

        # output
        X = np.array(
            [
                [[0, 0], [1, 1], [2, 2]],
                [[1, 1], [2, 2], [3, 3]],
                [[2, 2], [3, 3], [4, 4]],
                [[3, 3], [4, 4], [5, 5]],
                [[4, 4], [5, 5], [6, 6]],
                [[5, 5], [6, 6], [7, 7]],
                [[6, 6], [7, 7], [8, 8]],
            ]
        )

        y = np.array([[3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])

        result_X, result_y = transform_data(df, max, min, window_size)

        self.assertTrue((result_X == X).all())
        self.assertTrue((result_y == y).all())


if __name__ == "__main__":
    unittest.main()
