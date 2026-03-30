PID_WALK_OPD = [2e-5, 0.02, 0]
PID_TRACK_OPD = [2e-5, 2e-6, 0.]
PID_TRACK_DA1 = [-0.0003, -5e-5, 0.]
PID_TRACK_DA2 = [-0.0003, -5e-5, 0.]

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
