from setuptools import setup

setup(
    name='wotstomata',
    version='0.1',
    description=(
        'stomatal prediction using stacked hourglass neural net'
    ),
    author='Matthew Parker',
    scripts=[
        'wotstomata/scripts/train_stomatal_prediction'
    ],
    entry_points={
        'console_scripts': [
            'ws_train = wotstomata.scripts.train_hourglass:main',
            'ws_pred = wotstomata.scripts.predict_image:predict_image'
        ]
    },
    packages=[
        'wotstomata',
    ],
    include_package_data=True,
    install_requires=[
        'click',
        'scikit-image==0.14.dev0',
        'networkx',
        'pillow',
        'h5py',
        'numpy',
        'shapely',
        'rasterio',
        'keras',
        'matplotlib',
        'seaborn',
        'read_roi',
    ],
)
