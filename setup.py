from distutils.core import setup
import os
version_file = open(os.path.join(os.path.dirname(__file__), 'VERSION'))
version = version_file.read().strip()
setup(
    name = 'gpmhc',
    packages = ['gpmhc'],
    version = version,  # Ideally should be same as your GitHub release tag varsion
    description = 'Inference for graph-pmhc',
    author = 'Will Thrift, Quade Broadwell',
    author_email = 'thriftw@gene.com',
    url = '',
    download_url = '',
    keywords = [],  # Maybe add some keywords here if you want people to find it organically
    classifiers = [],
)
