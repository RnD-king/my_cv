from setuptools import find_packages, setup
import glob
import os

package_name = 'my_cv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.*'))),
        ('share/' + package_name + '/config', glob.glob(os.path.join('config', '*.yaml*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='noh',
    maintainer_email='noh@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_tracker1 = my_cv.line_tracker1:main',
            'line_tracker2 = my_cv.line_tracker2:main',
            'ball_detect = my_cv.ball_detect:main',
            'ball_recieve = my_cv.ball_recieve:main', 
            'realsense_test = my_cv.realsense_test:main',              
        ],
    },
)
