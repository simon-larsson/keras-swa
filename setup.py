from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='keras-swa',
    version='0.1.3',
    description='Simple stochastic weight averaging callback for Keras.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    author='Simon Larsson',
    author_email='simonlarsson0@gmail.com',
    url='https://github.com/simon-larsson/keras-swa',
    license='MIT',
    install_requires=[],
    classifiers=['License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 3 :: Only',
                 'Topic :: Scientific/Engineering']
)