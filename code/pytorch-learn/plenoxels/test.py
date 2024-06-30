import unittest

from unit import add


class PlenoxelTest(unittest.TestCase):
    def test_can_add(self):
        self.assertEqual(add(1, 2), 3)
