from setuptools import setup, find_packages
from pathlib import Path


def load_requirements(fname: Path):
    reqs = []
    with fname.open('r') as reqs_file:
        for line in reqs_file:
            reqs.append(line.rstrip())
    return reqs


setup(
    name='phewas',
    version='1.0.1',
    packages=find_packages(),
    url='https://github.com/mrcepid-rap/mrcepid-runassociationtesting-phewas',
    license='MIT',
    author='Eugene Gardner',
    author_email='eugene.gardner@mrc-epid.cam.ac.uk',
    description='',
    install_requires=load_requirements(Path('requirements.txt'))
)