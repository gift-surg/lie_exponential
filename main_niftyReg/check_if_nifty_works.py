"""
Aim is to use niftyReg from python.
See initially if it works!

"""

import os
from utils.path_manager import niftyReg_path


###


print 'Nifty Reg path: '
print niftyReg_path

path_reg_tool = os.path.join(niftyReg_path, 'reg_aladin')

os.system(path_reg_tool + ' -h')
