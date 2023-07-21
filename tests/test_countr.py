import os
import sys
import unittest
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from vision_counter.models.countr import CounTR


class TestCountR(unittest.TestCase):
    def setUp(self):
        # Initialize CountR with the model path (Update with your actual model path)
        self.model_path = sys.argv[1]
        self.count_r = CounTR(self.model_path)

    def test_count_with_sample_image(self):
        # Load a sample image and bounding boxes
        image_path = sys.argv[2]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = eval(sys.argv[3])

        # Perform counting using the count method
        count_result = self.count_r.count(image, bboxes)

        # Perform your assertions here to check the result
        # For example, you can check if the count_result is of the expected data type
        self.assertIsInstance(count_result, int)


if __name__ == '__main__':
    unittest.main()
