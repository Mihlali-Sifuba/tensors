import unittest

from tensors.utils.lists import flatten_nested_list, infer_nested_list_shape


class ListUtilityTests(unittest.TestCase):
    def test_flatten_nested_list_preserves_row_major_order(self):
        nested = [[1, 2], [[3], 4], 5]

        self.assertEqual(flatten_nested_list(nested), [1, 2, 3, 4, 5])

    def test_flatten_nested_list_handles_empty_lists(self):
        self.assertEqual(flatten_nested_list([[], [[], []]]), [])

    def test_flatten_nested_list_does_not_use_python_recursion(self):
        nested = [1]
        for _ in range(1500):
            nested = [nested]

        self.assertEqual(flatten_nested_list(nested), [1])

    def test_flatten_nested_list_rejects_a_cycle(self):
        nested = []
        nested.append(nested)

        with self.assertRaisesRegex(ValueError, "Cyclic nested lists"):
            flatten_nested_list(nested)

    def test_flatten_nested_list_allows_a_shared_non_cyclic_child(self):
        child = [1, 2]

        self.assertEqual(flatten_nested_list([child, child]), [1, 2, 1, 2])

    def test_infer_nested_list_shape_handles_rectangular_data(self):
        self.assertEqual(
            infer_nested_list_shape([[[1], [2]], [[3], [4]]]),
            (2, 2, 1),
        )

    def test_infer_nested_list_shape_handles_empty_dimensions(self):
        self.assertEqual(infer_nested_list_shape([]), (0,))
        self.assertEqual(infer_nested_list_shape([[], []]), (2, 0))

    def test_infer_nested_list_shape_treats_a_non_list_as_scalar(self):
        self.assertEqual(infer_nested_list_shape(3), ())

    def test_infer_nested_list_shape_rejects_different_child_lengths(self):
        with self.assertRaisesRegex(ValueError, "Ragged nested lists"):
            infer_nested_list_shape([[1, 2], [3]])

    def test_infer_nested_list_shape_rejects_mixed_scalars_and_lists(self):
        with self.assertRaisesRegex(ValueError, "Ragged nested lists"):
            infer_nested_list_shape([1, [2]])

    def test_infer_nested_list_shape_rejects_a_cycle(self):
        nested = []
        nested.append(nested)

        with self.assertRaisesRegex(ValueError, "Cyclic nested lists"):
            infer_nested_list_shape(nested)

    def test_infer_nested_list_shape_allows_a_shared_non_cyclic_child(self):
        child = [1, 2]

        self.assertEqual(infer_nested_list_shape([child, child]), (2, 2))


if __name__ == "__main__":
    unittest.main()
