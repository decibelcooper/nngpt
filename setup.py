from os import path
import setuptools

# read the contents of the README.md file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md')) as f:
    long_description = f.read()

setuptools.setup(
    name='nngpt',
    version='0.2',
    description='Toolkit for fast-ish nonnegative Gaussian process tomography',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://github.com/decibelcooper/nngpt',
    author='David Blyth',
    author_email='dblyth@decibelcooper.net',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'tensorflow',
        'scipy',
        'matplotlib',
        ],
    zip_safe=True,
)
