from collections import namedtuple

VER = namedtuple('VERSION', 'version date changes')

__version_list__ = [
    VER('0.1.0.dev1', '032917', 'Initial'),
    VER('0.1.0.dev2', '091217', 'Creating proper module split by concern')
]

__current_version__ = __version_list__[-1]
__version__ = __current_version__.version


