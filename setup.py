from setuptools import setup, find_packages

setup(
    name='page_segmentation',
    version='0.0.1',
    packages=find_packages(),
    license='GPL-v3.0',
    long_description=open("README.md").read(),
    include_package_data=True,
    author="Christoph Wick",
    author_email="christoph.wick@informatik.uni-wuerzburg.de",
    url="https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation.git",
    download_url='https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation.git',
    entry_points={
        'console_scripts': [
            'page-segmentation=pagesegmentation.scripts.pagesegmentation:main',
        ],
    },
    install_requires=open("requirements.txt").read().split(),
    keywords=['OCR', 'page segmentation', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
)
