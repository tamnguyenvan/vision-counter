# Vision Counter: Object Counting Library using Deep Learning
VisionCounter is a powerful Python library that provides a collection of deep learning models for accurately counting objects in input images. Whether you need to count people in a crowded street, cars in a parking lot, or any other objects in a scene, VisionCounter has got you covered.

<div style="text-align: center;">
  <img src="./images/logo.png" alt="logo" />
</div>

# Key Features
- State-of-the-Art Models: VisionCounter comes with pre-trained state-of-the-art deep learning models, trained on large-scale datasets, to deliver high accuracy in counting various objects.

- Easy-to-Use API: We believe in simplicity and ease of use. VisionCounter provides a straightforward API, allowing you to seamlessly integrate object counting capabilities into your Python projects.

- Customizable: While the default pre-trained models work well in many scenarios, we understand that every project is unique. VisionCounter allows you to fine-tune models or even train your custom models on your specific datasets.

- Fast and Efficient: Our models are designed for speed and efficiency, making them suitable for real-time applications or processing large batches of images.

- Wide Range of Applications: VisionCounter is versatile and can be used in various domains, including surveillance, traffic monitoring, inventory management, and more.

# Quick Start
Install the library using pip
```
pip install vision_counter
```

Usage
```
from sys import argv
from vision_counter import CounTR

counter = CounTR()
bboxes = [[[156, 112], [173, 129]]]  # x1, y1, x2, y2

image_path = argv[1]
object_count = counter.count(image_path, bboxes)
print("Estimated object count:", object_count)
```

# Contributing
We welcome contributions from the community! Whether it's bug fixes, feature enhancements, or new models, feel free to submit pull requests.

# License
VisionCounter is licensed under the MIT License. See the LICENSE file for details.

Feel free to customize and expand upon this introduction to suit the specifics of your VisionCounter library. Make sure to replace link_to_logo_image with the actual URL or path to your library's logo image, and provide more details about the models, datasets, and any other relevant information that makes your library stand out.