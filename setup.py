from setuptools import setup, find_packages

# Function to read requirements from requirements.txt file
def get_requirements():
    with open('requirements.txt') as f:
        return f.read().splitlines()

setup(
    name='vision_counter',
    version='0.1.0',
    description='Vision Counter: An Efficient Object Counting Library for Python',
    long_description='Vision Counter is a powerful and user-friendly Python library that enables accurate object counting in images using pre-trained models. With a simple and intuitive API, Vision Counter can be easily integrated into your computer vision projects, providing reliable object counting solutions for various applications. The library comes equipped with state-of-the-art models and efficient algorithms to deliver high-performance object counting capabilities. Whether you need to count people, vehicles, or any other objects in your images, Vision Counter is the ideal choice for fast and accurate results.',
    long_description_content_type='text/markdown',
    url='https://github.com/tamnguyenvan/vision-counter',
    author='Tam Nguyen',
    author_email='tamnvhustcc@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='computer-vision object-counting image-recognition deep-learning python-library',
    project_urls={
        'Documentation': 'https://tamnguyenvan.github.io/vision_counter',
        'Source': 'https://github.com/tamnguyenvan/vision_counter',
    },
)
