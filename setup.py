from setuptools import setup, find_packages

setup(
    name='rl_pendulum_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numba',
        'matplotlib'
    ],
    author='Scott Fortune',
    description='Custom RL environment and Evolution Strategy training for pendulum control.',
)
