from setuptools import find_packages, setup

setup(
    name='gato',
    version='0.0.1',
    description='Gato: A Generalist Agent',
    url='https://github.com/kyegomez/GATO',
    author='Kye Gomez',
    author_email='kye@apac.ai',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(exclude=[]),
    install_requires=[
        'tensorflow>=2.11',
        'flowchain>=0.0.4'
    ],
    python_requires='>=3.10.0',
    keywords=[
        'deep learning',
        'gato',
        'tensorflow',
        'generalist agent'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10'
    ]
)
