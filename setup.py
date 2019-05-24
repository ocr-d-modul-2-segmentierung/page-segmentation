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
            'page-segmentation=pagesegmentation.scripts.main:main',
            'ocrd-compute-normalizations=pagesegmentation.scripts.compute_image_normalizations:main',
            'ocrd-pixel-classifier=pagesegmentation.scripts.predict:main',
        ],
    },
    install_requires=open("requirements.txt").read().split(),
    extras_require={
        'tf_cpu': ['tensorflow>=1.6.0'],
        'tf_gpu': ['tensorflow-gpu>=1.6.0'],
    },
    keywords=['OCR', 'page segmentation', 'pixel classifier'],
    data_files=[('', ["requirements.txt"])],
)
