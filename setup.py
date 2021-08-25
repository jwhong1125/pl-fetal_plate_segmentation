
import sys
import os


# Make sure we are running python3.5+
if 10 * sys.version_info[0] + sys.version_info[1] < 35:
    sys.exit("Sorry, only Python 3.5+ is supported.")


from setuptools import setup


def readme():
    print("Current dir = %s" % os.getcwd())
    print(os.listdir())
    with open('README.rst') as f:
        return f.read()

setup(
      name             =   'fetal_plate_segmentation',
      # for best practices make this version the same as the VERSION class variable
      # defined in your ChrisApp-derived Python class
      version          =   '0.1',
      description      =   'An app to segment the cortical plate of fetal T2 MRI using deep leraning.',
      long_description =   readme(),
      author           =   'Jinwoo Hong',
      author_email     =   'Jinwoo.Hong@childrens.harvard.edu',
      url              =   'http://wiki',
      packages         =   ['fetal_plate_segmentation'],
      install_requires =   ['chrisapp', 'pudb', 'tensorflow==2.5.1', 'keras==2.2.4', 'matplotlib'],
      test_suite       =   'nose.collector',
      tests_require    =   ['nose'],
      scripts          =   ['fetal_plate_segmentation/fetal_plate_segmentation.py'],
      license          =   'MIT',
      zip_safe         =   False
     )
