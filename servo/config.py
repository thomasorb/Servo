import numpy as np

#ROI_SHAPE = np.array((32, 32), dtype=int)
#ROI_SIZE = np.prod(ROI_SHAPE)

IRCAM_DEFAULT_EXPOSURE_TIME = '1us'
IRCAM_SERVO_OUTPUT_TIME = 0.001 #s
IRCAM_VIEWER_OUTPUT_TIME = 0.1 #s
IRCAM_BUFFER_SIZE = 10000

FRAME_DTYPE = np.float32
DATA_DTYPE = np.float32

FULL_FRAME_SHAPE = np.array((320, 256), dtype=int)
FULL_FRAME_SIZE = np.prod(FULL_FRAME_SHAPE)
FULL_FRAME_CENTER = np.array((FULL_FRAME_SHAPE[0]//2, FULL_FRAME_SHAPE[1]//2), dtype=int)
DEFAULT_FRAME_POSITION = np.array((0, 0), dtype=int) # first column, first line
MIN_ROI_SHAPE = 32

DAQ_PIEZO_LEVELS_DTYPE = np.float32
DAQ_PIEZO_CHANNELS = [0, 2, 4] # OPD, DA-1, DA-2
DAQ_CHANGE_SPEED = 5 # level change per second
DAQ_LOOP_TIME = 0.01
DAQ_MAX_LEVEL_CHANGE = DAQ_CHANGE_SPEED * DAQ_LOOP_TIME


OPD_LOOP_TIME = 0.1  # 10 Hz loop
PIEZO_V_MIN = 0.0
PIEZO_V_MAX = 10.0
OPD_TOLERANCE = 5.0  # nm
PIEZO_MAX_OPD_DIFF = 5000 # nm

BUFFER_SIZE = 100 # for servo values buffering
VIEWER_BUFFER_SIZE = 1000 # for viewer servo values buffering

VIEWER_ELLIPSE_DRAW_BUFFER_SIZE = 100

DEFAULT_PROFILE_LEN = MIN_ROI_SHAPE
DEFAULT_PROFILE_WIDTH = 4
DEFAULT_PROFILE_POSITION = np.array((DEFAULT_PROFILE_LEN//2, DEFAULT_PROFILE_LEN//2), dtype=int)

DEFAULT_PID = [1e-3, 0., 0.]

SERVO_EVENTS = (
    'normalize',
    'stop',
    'start',
    'move_to_opd',
    'close_loop',
    'open_loop',
    'roi_mode',
    'full_frame_mode',
    'reset_zpd',
)

NEXLINE_EVENTS = (
    'start',
    'move',
    'stop_move',
    )

# use only this part of the profiles for normalization coeffs computation
NORMALIZATION_LEN_RATIO = 0.7


CALIBRATION_LASER_WAVELENGTH = 1550 # nm
LASER_ANGLE = 25 # angle of the laser in degrees
NEXLINE_CHANNEL = 1
NEXLINE_STEP_SIZE = 5 # um in mechanical path difference
NEXLINE_TIMEOUT = 300 # s
