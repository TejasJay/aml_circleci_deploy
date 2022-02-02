import unittest

from app import app
import os


class TestToPerform(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_page(self): # if root url gives successful response it means the app is running fine and passed the test case
        response = self.app.get('/', follow_redirects=True) 
        print(response)
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
