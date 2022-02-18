from distutils.core import setup

setup(name="LANDMark",
                 version="1.0.0.dev2",
                 author="Josip Rudar, Teresita M. Porter, Michael Wright, Brian Golding, Mehrdad Hajibabaei",
                 author_email="rudarj@uoguelph.ca",
                 description="LANDMark: An ensemble approach to the supervised selection of biomarkers in high-throughput sequencing data",
                 url="https://github.com/jrudar/LANDMark",
                 license = "MIT",
                 keywords = "biomarker selection, metagenomics, metabarcoding, biomonitoring, ecological assessment, machine learning, supervised learning, unsupervised learning",
                 packages=["LANDMark"],
                 python_requires = ">=3.8",
                 install_requires = ["scikit-learn >= 0.23",
                                     "joblib >= 0.17.0",
                                     "tensorflow_addons >= 0.12.0",
                                     "numpy >= 1.18",
                                     "pandas >= 1.0.3",
                                     "numba >= 0.44.1",
                                     "shap >= 0.34.0"],
                 classifiers=["Programming Language :: Python :: 3.8+",
                              "License :: MIT License",
                              "Operating System :: OS Independent",
                              "Topic :: Ecology :: Biomarker Selection :: Metagenomics :: Supervised Learning :: Unsupervised Learning :: Metabarcoding :: Biomonitoring :: Ecological Assessment :: Machine Learning"],
                 )