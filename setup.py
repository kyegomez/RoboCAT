from setuptools import setup, find_packages

setup(
  name = 'robocat',
  packages = find_packages(exclude=[]),
  version = '0.0.1',
  license='MIT',
  description = 'Robo CAT- Pytorch',
  author = 'Kye Gomez',
  author_email = 'kye@apac.ai',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/kyegomez/RoboCAT',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'robotics'
  ],
  install_requires=[
    'classifier-free-guidance-pytorch>=0.1.4',
    'einops>=0.6',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)