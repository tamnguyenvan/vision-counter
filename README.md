![GitHub license](https://img.shields.io/github/license/tamnguyenvan/vision-counter)
![GitHub top language](https://img.shields.io/github/languages/top/tamnguyenvan/vision-counter)
![GitHub stars](https://img.shields.io/github/stars/tamnguyenvan/vision-counter)

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
```python
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

# Citation
```
@article{liu2022countr,
  author = {Chang, Liu and Yujie, Zhong and Andrew, Zisserman and Weidi, Xie},
  title = {CounTR: Transformer-based Generalised Visual Counting},
  journal = {arXiv:2208.13721},
  year = {2022}
}
```