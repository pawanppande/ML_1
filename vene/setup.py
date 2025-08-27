from setuptools import setup, find_packages
from typing import List
def get_requirements(file_path:str)->list:
   ''' this function will return the list of requirements 
   mentioned in the requirements.txt file'''
   with open(file_path) as file_obj:
       requirement = file_obj.readlines()
       requirement = [req.replace("\n", "") for req in requirement]
       
       if '-e .' in requirement:
           requirement.remove('-e .')
    return requirement

setup(
    name = 'mlproject',
    version = '0.0.1',
    author= 'Pawan Pande',
    author_email= 'pawanppande1496@gmail.com'
    packages = find_packages(),
    install_requires = get_requirements('requirement.txt'

)
