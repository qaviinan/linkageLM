from setuptools import setup, find_packages

setup(
    name='linkage_lm',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'pandas==1.3.3',
        'recordlinkage==0.14',
        'embedchain==0.1.0',
        'llama3api==0.2.1',
        'openai==0.27.0',
        'requests==2.26.0',
        'beautifulsoup4==4.10.0',
        'langchain==0.0.1',  # Adjust version as needed
        'pydantic==1.8.2',
        'python-dotenv==0.19.2'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A module for fuzzy matching using various techniques.',
    url='https://github.com/yourusername/linkage_lm',  # Update with your repo URL
)
