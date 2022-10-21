from setuptools import find_packages, setup

setup(
    name="doors",
    author="Matias Thayer",
    author_email="matias.thayer@gmail.com",
    url="https://github.com/chechir/doors",
    version="0.0.3",
    license="MIT",
    setup_requires=["setuptools_scm"],
    packages=find_packages(exclude=["tests"]),
    install_requires=("numexpr", "pandas", "scikit-learn", "scipy", "numpy",),
)
