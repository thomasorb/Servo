PROFILE_WIDTH = 4 # pixels, width of the profile used for servoing

PID_WALK_OPD = [1e-4, 2e-7, 0]
PID_TRACK_OPD = [2e-5, 0, 0.]
PID_TRACK_DA1 = [-0.0003, 0., 0.]
PID_TRACK_DA2 = [-0.0003, 0., 0.]
PID_WALK_DA1 = [0, -5e-7, 0.]
PID_WALK_DA2 = [0, -5e-7, 0.]

PID_DA_DEADBAND = 0.02
PID_DA_KAW = 0.

NEXLINE_CALIB_OPD = 30000 # nm, opd diplacement to reach for nexline velocity calibration
NEXLINE_OPD_UPDATE = 30000 # nm, opd displacement to update nexline velocity

PIEZO_DA_LOOP_UPDATE_TIME = 1.0 # s, time to update new DA base values
PIEZO_DA_LOOP_MAX_V_DIFF = 0.05 # max V diff when looping to avoid lost

PIEZO_OPD_STEP = 0.1 # nm
PIEZO_OPD_DIRAC = 1.0 # nm
PIEZO_DA_STEP = 0.01 # nm
PIEZO_DA_DIRAC = 0.04 # nm

FIR_GAIN_OPD = 1.
FIR_GAIN_DA1 = 1.
FIR_GAIN_DA2 = 1.
FIR_SCALE_OPD = 1e-9 # 1 nm
FIR_SCALE_ANG = 1e-6 # 1 urad
FIR_LMS_MU = 0.01

NEXLINE_VELOCITY_ADJUSTMENT_GAIN = 1.3 # must be > 1 to move back the piezo near the 5 V range.

IA_TRAINING_SCALE_OPD = 0.001 # scale for piezo random walk during IA training
IA_TRAINING_SCALE_DA = 0.01 # scale for piezo random walk during IA training

SERVO_WALK_NORMALIZE_TIME = 15.0 # s, time to record values for normalization coeffs computation during walk to opd
SERVO_WAIT_NORMALIZE_TIME = 15.0 # s, time to record values for normalization coeffs computation during walk to opd
SERVO_DA_LOOP_ENABLED = True # whether to enable the DA loop when servoing to OPD

WAITING_PIEZO_PERIOD = 60 # s, period of the piezo in waiting mode
