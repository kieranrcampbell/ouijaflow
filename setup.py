
from setuptools import setup

setup(name='ouijaflow',
      version='0.1',
      description='Ouija in Edward and Tensorflow',
      url='http://www.github.com/kieranrcampbell/ouijaflow',
      packages=['ouijaflow'],
      install_requires=[
          'tensorflow',
          'edward'
      ],
      zip_safe=False)
