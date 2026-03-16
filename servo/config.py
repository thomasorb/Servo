import numpy as np

CALIBRATION_LASER_WAVELENGTH = 1550 # nm
LASER_ANGLE = 25 # angle of the laser in degrees

#ROI_SHAPE = np.array((32, 32), dtype=int)
#ROI_SIZE = np.prod(ROI_SHAPE)

IRCAM_DEFAULT_EXPOSURE_TIME = '1us'
IRCAM_SERVO_OUTPUT_TIME = 0.005 #s
IRCAM_VIEWER_OUTPUT_TIME = 0.1 #s

IRCAM_BUFFER_SIZE = 100
IRCAM_LOST_THRESHOLD = CALIBRATION_LASER_WAVELENGTH / 2 * 0.8 # nm (80% of lambda/2 to be safe)

FRAME_DTYPE = np.float32
DATA_DTYPE = np.float32

FULL_FRAME_SHAPE = np.array((320, 256), dtype=int)
FULL_FRAME_SIZE = np.prod(FULL_FRAME_SHAPE)
FULL_FRAME_CENTER = np.array((FULL_FRAME_SHAPE[0]//2, FULL_FRAME_SHAPE[1]//2), dtype=int)
DEFAULT_FRAME_POSITION = np.array((0, 0), dtype=int) # first column, first line
MIN_ROI_SHAPE = 32

DAQ_PIEZO_LEVELS_DTYPE = np.float32
DAQ_PIEZO_CHANNELS = [0, 4, 2] # OPD, DA-1, DA-2
DAQ_CHANGE_SPEED = 10 # level change per second
DAQ_LOOP_TIME = 0.01
DAQ_MAX_LEVEL_CHANGE = DAQ_CHANGE_SPEED * DAQ_LOOP_TIME
DAQ_PIEZO_OPD_PER_LEVEL = 2500 # nm/level (roughly)

OPD_LOOP_TIME = 0.01  # 100 Hz loop
PIEZO_V_MIN = 0.0
PIEZO_V_MAX = 10.0
OPD_TOLERANCE = 5.0  # nm
PIEZO_MAX_OPD_DIFF = 5000 # nm

SERVO_BUFFER_SIZE = 20 # for servo values buffering: servo updates are based on mean of this buffer
SERVO_NORMALIZE_REC_TIME = 1.0 # s, time to record values for normalization coeffs computation
SERVO_NORMALIZE_REC_SIZE = 10000 # s, time to record values for normalization coeffs computation
VIEWER_BUFFER_SIZE = 1000 # for viewer servo values buffering
SERVO_DEFAULT_NICENESS = 0
SERVO_MAX_NICENESS = -20
SERVO_LOW_NICENESS = 10
SERVO_OPD_TIMEOUT = 5.0 # s
SERVO_NONCRITIC_REFRESH_TIME = 1 # s, time to refresh servo config (PID coeffs, etc.)

VIEWER_ELLIPSE_DRAW_BUFFER_SIZE = 100

DEFAULT_PROFILE_LEN = MIN_ROI_SHAPE
DEFAULT_PROFILE_WIDTH = 4
DEFAULT_PROFILE_POSITION = np.array((DEFAULT_PROFILE_LEN//2, DEFAULT_PROFILE_LEN//2), dtype=int)

SERIAL_PORT = "/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller-if00-port0"
SERIAL_BAUDRATE = 9600 # bps
SERIAL_STATUS_RATE = 100 # Hz
SERIAL_STATUS_FRAME_SIZE = 9 # number of bytes in the status frame (header 3 + payload 5 + checksum 1)


TRACKER_BUFFER_SIZE = 10000
TRACKER_FREQUENCY = 100 # Hz
TRACKER_STATS_FREQUENCIES = [1, 3, 10, 30, 100] # Hz

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
    'walk_to_opd',
    'calibrate_velocity',
)

NEXLINE_EVENTS = (
    'start',
    'move',
    'stop_move',
    )

# use only this part of the profiles for normalization coeffs computation
NORMALIZATION_LEN_RATIO = 0.7

NEXLINE_CHANNEL = 1
NEXLINE_MOVING_VELOCITY = 20 # um/s (optical)
NEXLINE_STEP_SIZE = 5 # um in mechanical path difference
NEXLINE_TIMEOUT = 300 # s
NEXLINE_MIN_VELOCITY = 0.001
NEXLINE_MAX_VELOCITY = 50
NEXLINE_POS_CALIB_FACTOR = 1.
NEXLINE_NEG_CALIB_FACTOR = 1.
