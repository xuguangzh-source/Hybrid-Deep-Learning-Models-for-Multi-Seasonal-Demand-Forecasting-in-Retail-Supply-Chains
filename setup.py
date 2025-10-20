from setuptools import setup, find_packages

setup(
    name="hybrid_demand_forecasting",
    version="0.1.0",
    description="Hybrid (TCN + BiLSTM + Attention) deep learning for multi-seasonal retail demand forecasting",
    author="Xuguang Zhang",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.9",
)
