import unittest
from subaligner.singleton import Singleton


class SingletonTests(unittest.TestCase):
    def test_singleton(self):
        class Single(metaclass=Singleton):
            pass

        a = Single()
        b = Single()
        self.assertEqual(a, b)
        a.attribute = "attribute"
        self.assertTupleEqual(("attribute", "attribute"), (a.attribute, b.attribute))


if __name__ == "__main__":
    unittest.main()
