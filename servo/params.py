PID_WALK_OPD = [1e-3, 0.1, 1e-4]
PID_TRACK_OPD = [2e-4, 5e-6, 0.]
PID_TRACK_DA1 = [-0.03, -0.0001, 0.]
PID_TRACK_DA2 = [-0.03, -0.0001, 0.]

NEXLINE_CALIB_OPD = 5000 # nm, opd diplacement to reach for nexline velocity calibration
NEXLINE_OPD_UPDATE = 10000 # nm, opd displacement to update nexline velocity

PIEZO_DA_LOOP_UPDATE_TIME = 1.0 # s, time to update new DA base values
PIEZO_DA_LOOP_MAX_V_DIFF = 0.05 # max V diff when looping to avoid lost

PIEZO_OPD_STEP = 0.1 # nm
PIEZO_OPD_DIRAC = 1.0 # nm
PIEZO_DA_STEP = 0.01 # nm
PIEZO_DA_DIRAC = 0.04 # nm

