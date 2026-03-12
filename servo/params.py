PID_OPD = [2e-6, 0., 0.]
PID_DA = [-0.003, -0.0001, 0.]
PID_NEXLINE = [0.1, 0., 0.]
NEXLINE_CALIB_OPD = 5000 # nm, opd diplacement to reach for nexline velocity calibration
NEXLINE_OPD_UPDATE = 10000 # nm, opd displacement to update nexline velocity

PIEZO_DA_LOOP_UPDATE_TIME = 1.0 # s, time to update new DA base values
PIEZO_DA_LOOP_MAX_V_DIFF = 0.05 # max V diff when looping to avoid lost

PIEZO_OPD_STEP = 0.1 # nm
PIEZO_OPD_DIRAC = 1.0 # nm
PIEZO_DA_STEP = 0.01 # nm
PIEZO_DA_DIRAC = 0.04 # nm

