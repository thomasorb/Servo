import numpy as np

ROI_SHAPE = np.array((32, 32), dtype=int)
ROI_SIZE = np.prod(ROI_SHAPE)
FRAME_DTYPE = np.float32
FULL_FRAME_SHAPE = np.array((320, 256), dtype=int)
FULL_FRAME_SIZE = np.prod(FULL_FRAME_SHAPE)
FULL_FRAME_CENTER = np.array((FULL_FRAME_SHAPE[0]//2, FULL_FRAME_SHAPE[1]//2), dtype=int)
DEFAULT_ROI_POSITION = np.array((0, 0), dtype=int) # first column, first line
MIN_ROI_SHAPE = 32

DAQ_PIEZO_LEVELS_DTYPE = np.float32
DAQ_PIEZO_CHANNELS = [0, 2, 4] # OPD, DA-1, DA-2
DAQ_CHANGE_SPEED = 1.0 # level change per second
DAQ_LOOP_TIME = 0.1
DAQ_MAX_LEVEL_CHANGE = DAQ_CHANGE_SPEED * DAQ_LOOP_TIME




