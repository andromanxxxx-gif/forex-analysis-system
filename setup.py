from setuptools import setup, find_packages

setup(
    name="forex-analysis-app",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "flask==2.3.3",
        "pandas==2.0.3",
        "numpy==1.24.3",
        "requests==2.31.0",
        "python-dotenv==1.0.0",
    ],
    python_requires=">=3.8",
)
