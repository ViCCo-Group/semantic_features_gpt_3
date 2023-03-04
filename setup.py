from setuptools import setup, find_packages
setup(
    name='gptfeatures',
    version='1.0',
    description='Python Distribution Utilities',
    author='Hannes Hansen',
    author_email='gward@python.net',
    scripts = ['bin/decode.py', 'bin/encoding.py'],
    packages=['utils', 'utils.analyses', 'utils.decoding', 'utils.encoding'],
    install_requires=[],
)