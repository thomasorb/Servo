import numpy as np
import time
import logging
import traceback
import collections

from . import NITLibrary_x64_382_py312 as NITLibrary

from . import core
from . import config
from . import utils

log = logging.getLogger(__name__)

class IRCamera(core.Worker):

    def __init__(self, data, events, frame_shape=None, frame_center=None, roi_mode=False):
        
        super().__init__(data, events)

        self.roi_mode = bool(roi_mode)
        
        if frame_shape is None:
            self.frame_shape = np.array(config.FULL_FRAME_SHAPE)
        else:
            self.frame_shape = np.array(frame_shape)
            
        if frame_center is None:
            self.frame_position = np.array(config.DEFAULT_FRAME_POSITION)
        else:
            self.frame_position = np.array(frame_center) - self.frame_shape//2


        self.requested_frame_shape = np.copy(self.frame_shape)
        self.requested_frame_position = np.copy(self.frame_position)

        # set frame shape and positions to camera accepted values
        self.frame_shape = self.frame_shape // 8 * 8 # shape must be a multiple of 8
        self.frame_position = self.frame_position // 4 * 4 # position must be a multiple of 4

        if np.any(self.requested_frame_shape != self.frame_shape):
            log.warning(f'frame shape {self.requested_frame_shape} was changed to {self.frame_shape} to fit camera requirements')
        if np.any(self.requested_frame_position != self.frame_position):
            log.warning(f'frame position {self.requested_frame_position} was changed to {self.frame_position} to fit camera requirements')
            
            
        assert (0 <= self.frame_position[0] < config.FULL_FRAME_SHAPE[0]), f'bad x position: {self.frame_position[0]}'
        assert (0 <= self.frame_position[1] < config.FULL_FRAME_SHAPE[1]), f'bad y position: {self.frame_position[1]}'

        assert (config.MIN_ROI_SHAPE <= self.frame_shape[0] <= config.FULL_FRAME_SHAPE[0]), f'bad shape: {self.frame_shape[0]}'
        assert (config.MIN_ROI_SHAPE <= self.frame_shape[1] <= config.FULL_FRAME_SHAPE[1]), f'bad shape: {self.frame_shape[1]}'

        log.info(f'target FRAME {self.frame_shape} at {self.frame_position}')

        self.frame_size = int(np.prod(self.frame_shape))
        self.data['IRCamera.frame_size'][0] = int(self.frame_size)
        self.data['IRCamera.frame_dimx'][0] = int(self.frame_shape[0])
        self.data['IRCamera.frame_dimy'][0] = int(self.frame_shape[1])
        
        self.data = data
        
        self.nm = NITLibrary.NITManager.getInstance() #Get unique instance of NITManager
        log.info(self.nm.listDevices())
        self.dev = self.nm.openOneDevice()
        assert self.dev is not None, log.error("IR Camera not connected")
        
        self.config_obs = ConfigObserver()
        self.dev << self.config_obs
        
        # Device Configuration

        # #Get param value (2 ways)
        # print("Exposure:" + self.dev.paramStrValueOf( "Exposure Time" ) )
        # print("PixelClock:" + str( self.dev.paramValueOf( "Pixel Clock" ) ) )
        # #print("PixelDepth: " + self.dev.paramStrValueOf( NITLibrary.NITCatalog.PIX_DEPTH ))


        # Set params value
        self.dev.updateConfig()
        self.dev.setParamValueOf("Mode", "High Speed").updateConfig()
        self.dev.setParamValueOf("Sensor Response", 1).updateConfig()
        self.dev.setParamValueOf("Analog Gain", "Low").updateConfig()
        
        self.dev.setParamValueOf("Number Of Columns", int(self.frame_shape[0]))
        self.dev.setParamValueOf("Number Of Lines", int(self.frame_shape[1]))
        self.dev.setParamValueOf("First Column", int(self.frame_position[0]))
        self.dev.setParamValueOf("First Line", int(self.frame_position[1]))
        
        self.dev.setParamValueOf("ExposureTime", config.IRCAM_DEFAULT_EXPOSURE_TIME)
        self.dev.updateConfig()

        # Fps configuration: set fps to a mean value
        min_fps = self.dev.minFps()
        max_fps = self.dev.maxFps()
        log.info(f"fps range: ({self.dev.minFps()} - {self.dev.maxFps()})")
        self.dev.setFps(max_fps)
        self.dev.updateConfig()  #Data is sent to the device
        log.info(f"current fps: {self.dev.fps()}")

        # Set pipeline 
        #self.agc = NITLibrary.NITToolBox.NITAutomaticGainControl()
        #self.player = NITLibrary.NITToolBox.NITPlayer("Player")
        #self.dev << self.agc #<< self.player
        self.data_observer = DataObserver(self.data, self.frame_position, self.frame_shape,
                                          self.roi_mode)
        self.dev << self.data_observer
        self.data['IRCamera.initialized'][0] = True
        
        self.dev.start()	    #Start Capture
        log.info('capture started')
        
        
    def loop_once(self):
        time.sleep(0.1)

    def stop(self):
        self.dev.stop()      #Stop Capture
        
        log.info('capture stopped')

        

class ConfigObserver(NITLibrary.NITConfigObserver):
    def onParamChanged(self, param_name, str_value, num_value):
        log.debug("onParamChanged(" + param_name + ", \"" + str_value + "\", "
                     + str(num_value) + ")")
        
    def onParamRangeChanged(self, param_name, str_values, num_values, array_size,
                            cur_str_val, cur_num_val ):
        log.debug("onParamRangeChanged(" + param_name + ")")
        log.debug("\tcurrent = \"" + cur_str_val + "\", " + str(cur_num_val))
        if(str_values):
            log.debug("\tstr range = [\"" + str_values[0] + "\", \"" + str_values[-1] + "\"]" ) 
            log.debug("\tnum range = [" + str(num_values[0]) + ", "
                         + str(num_values[-1]) + "]" )
            
    def onFpsChanged(self, new_fps):
        log.debug("onFpsChanged(" + str(new_fps) + ")")
        
    def onFpsRangeChanged(self, new_fpsMin, new_fpsMax, new_fps ):
        log.debug("onFpsRangeChanged(" + str(new_fpsMin) + ", "
                     + str(new_fpsMax) + ", "  + str(new_fps) + ")")
        
    def onNewFrame(self, status):
        #log.debug("onNewFrame(" + str(status) + ")")
        pass
    
    def onNucChanged(self, nuc_str, status ):
        log.debug("onNucChanged(" + nuc_str + ", " + str(status) + ")")

# class DataFilter(NITLibrary.NITUserFilter):
#     def onNewFrame(self, frame):   #You MUST define this function - It will be called on each frames
#         if(frame.pixelType() == NITLibrary.ePixType.FLOAT ):
#             new_array = 1.0 - frame.data() #frame.data() gives access to the first pixel
#         else :
#             new_array = np.invert(frame.data()) 
#         return new_array        #Don't forget to return the resulting array

class DataObserver(NITLibrary.NITUserObserver):

    
    def __init__(self, data, frame_position, frame_shape, roi_mode=False):

        super().__init__()
        self.roi_mode = bool(roi_mode)
        self.data = data
        self.frame_position = frame_position
        self.frame_shape = frame_shape
        self.frame_size = int(np.prod(self.frame_shape))
        self.times = np.full(config.IRCAM_BUFFER_SIZE, np.nan, dtype=float)
        self.ids = np.full_like(self.times, np.nan)
        self.last_viewer_out_time = 0
        self.last_servo_out_time = 0
        self.stats_last_index = 0
        self.x_pixels_states = None
        self.y_pixels_states = None
        self.xpixels_list = None
        self.ypixels_list = None
        self.xpixels_list_pos = None
        self.ypixels_list_pos = None
        self.opd_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.tip_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.tilt_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        
    def onNewFrame(self, frame):
        try:
            index = int(frame.id()) - 1
            frame_time = time.time()
            self.times[index % self.times.size] = frame_time
            self.ids[index % self.times.size] = index

            # get stats when buffer fulled
            if index > self.stats_last_index + self.times.size: # buffers were fulled
                self.stats_last_index = (index // self.times.size) * self.times.size               
                diff_ids = np.diff(self.ids)
                mean_sampling_time = float(
                    (np.nanmax(self.times) - np.nanmin(self.times)) / np.sum(
                        ~np.isnan(self.times)))
                self.data['IRCamera.median_sampling_time'][0] = mean_sampling_time
                self.data['IRCamera.lost_frames'][0] = int(np.sum(diff_ids > 1))
                self.times.fill(np.nan)
                self.ids.fill(np.nan)

            # viewer output
            if frame_time - self.last_viewer_out_time > config.IRCAM_VIEWER_OUTPUT_TIME:
                self.last_viewer_out_time = frame_time
                self.data['IRCamera.last_frame'][:self.frame_size] = frame.data().T.flatten()

            # check if we want a full servo output or only fast opd computation
            # to detect opd loss
            if frame_time - self.last_servo_out_time > config.IRCAM_SERVO_OUTPUT_TIME:
                self.last_servo_out_time = frame_time
                compute_servo_output = True
            else:
                compute_servo_output = False

            try:
                # profile x and y always set in full frame coordinates
                if self.roi_mode:
                    ix, iy = self.frame_shape//2
                    profile_len = min(self.frame_shape)
                else:
                    ix = self.data['IRCamera.profile_x'][0] - self.frame_position[0]
                    iy = self.data['IRCamera.profile_y'][0] - self.frame_position[1]
                    profile_len = self.data['IRCamera.profile_len'][0]
                    
                profile_width = self.data['IRCamera.profile_width'][0]
                ilen = int(profile_len)
                if ilen > np.min(self.frame_shape): ilen = np.min(self.frame_shape)
                iwid = int(profile_width)

                try:
                    hprofile, vprofile, roi = utils.compute_profiles(frame.data().T, ix, iy,
                                                                     iwid, ilen,
                                                                     get_roi=True)

                except Exception as e:
                    log.error(f'Error at computing profiles on camera {traceback.format_exc()}')
                    return
                
                
                if compute_servo_output:
                    self.data['IRCamera.hprofile'][:profile_len] = hprofile
                    self.data['IRCamera.vprofile'][:profile_len] = vprofile    
                    self.data['IRCamera.roi'][:profile_len**2] = roi.flatten()

              
                # compute levels
                if compute_servo_output:

                    # get updated data and output normalized profiles
                    self.hnorm_min = np.ascontiguousarray(
                        self.data['Servo.hnorm_min'][:profile_len], dtype=config.DATA_DTYPE)
                    self.hnorm_max = np.ascontiguousarray(
                        self.data['Servo.hnorm_max'][:profile_len], dtype=config.DATA_DTYPE)
                    self.vnorm_min = np.ascontiguousarray(
                        self.data['Servo.vnorm_min'][:profile_len], dtype=config.DATA_DTYPE)
                    self.vnorm_max = np.ascontiguousarray(
                        self.data['Servo.vnorm_max'][:profile_len], dtype=config.DATA_DTYPE)

                    x_pixels_states = self.data['Servo.pixels_x']
                    y_pixels_states = self.data['Servo.pixels_y']
                    if self.x_pixels_states is None or not np.array_equal(
                            self.x_pixels_states, x_pixels_states):
                        self.x_pixels_states = np.copy(x_pixels_states)
                        self.xpixels_list = utils.get_pixels_lists(x_pixels_states)
                        self.xpixels_list_pos = utils.get_mean_pixels_positions(self.xpixels_list)
                        self.data['IRCamera.hprofile_levels_pos'][:3] = self.xpixels_list_pos

                    if self.y_pixels_states is None or not np.array_equal(
                            self.y_pixels_states, y_pixels_states):
                        self.y_pixels_states = np.copy(y_pixels_states)
                        self.ypixels_list = utils.get_pixels_lists(y_pixels_states)
                        self.ypixels_list_pos = utils.get_mean_pixels_positions(self.ypixels_list)
                        self.data['IRCamera.vprofile_levels_pos'][:3] = self.ypixels_list_pos

                    self.data['IRCamera.hprofile_normalized'][:profile_len] = utils.normalize_profile(hprofile, self.hnorm_min, self.hnorm_max, inplace=False)
                    self.data['IRCamera.vprofile_normalized'][:profile_len] = utils.normalize_profile(vprofile, self.vnorm_min, self.vnorm_max, inplace=False)


                # directly normalize and compute levels             
                hlevels = np.ascontiguousarray(np.empty(3, dtype=config.DATA_DTYPE))
                for i in range(3):
                    hlevels[i] = utils.normalize_and_compute_profile_level(
                        hprofile, self.hnorm_min, self.hnorm_max,
                        self.xpixels_list[i])

                vlevels = np.ascontiguousarray(np.empty(3, dtype=config.DATA_DTYPE))
                for i in range(3):
                    vlevels[i] = utils.normalize_and_compute_profile_level(
                        vprofile, self.vnorm_min, self.vnorm_max,
                        self.ypixels_list[i])

                if compute_servo_output:
                    self.data['IRCamera.hprofile_levels'][:3] = hlevels
                    self.data['IRCamera.vprofile_levels'][:3] = vlevels

                # compute angles and opd
                last_angles = np.ascontiguousarray(self.data['IRCamera.last_angles'][:4],
                                                   dtype=config.DATA_DTYPE)
                angles = np.ascontiguousarray(np.empty_like(last_angles))

                if compute_servo_output:
                    self.hellipse_norm_coeffs = np.ascontiguousarray(
                        self.data['Servo.hellipse_norm_coeffs'][:4], dtype=config.DATA_DTYPE)
                    self.vellipse_norm_coeffs = np.ascontiguousarray(
                        self.data['Servo.vellipse_norm_coeffs'][:4], dtype=config.DATA_DTYPE)
                    

                angles[:2] = utils.compute_angles(
                    hlevels, self.hellipse_norm_coeffs, last_angles[:2])
                
                angles[2:] = utils.compute_angles(
                    vlevels, self.vellipse_norm_coeffs, last_angles[2:])
                opds = utils.compute_opds(angles)

                opds -= self.data['IRCamera.mean_opd_offset']

                mean_opd = utils.mean(opds)
                
                
                if compute_servo_output:
                    self.data['IRCamera.opds'][:4] = opds.astype(config.FRAME_DTYPE)
                    self.data['IRCamera.mean_opd'][0] = float(mean_opd)

                    self.opd_deque.appendleft(float(mean_opd))
                    self.data['IRCamera.mean_opd_buffer'][:min(
                        len(self.opd_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                            self.opd_deque, dtype=config.FRAME_DTYPE)

                    tip = angles[1] - angles[0]
                    tilt = angles[3] - angles[2]
                    self.data['IRCamera.tip'][0] = tip
                    self.data['IRCamera.tilt'][0] = tilt

                    self.tip_deque.appendleft(float(tip))
                    self.data['IRCamera.tip_buffer'][:min(
                        len(self.tip_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                            self.tip_deque, dtype=config.FRAME_DTYPE)

                    self.tilt_deque.appendleft(float(tilt))
                    self.data['IRCamera.tilt_buffer'][:min(
                        len(self.tilt_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                            self.tilt_deque, dtype=config.FRAME_DTYPE)

                self.data['IRCamera.last_angles'][:4] = angles
                
                
                        
            except Exception as e:
                log.error(f'Error at reading profiles on camera {traceback.format_exc()}')
        
        except Exception as e:
            log.error('error on new frame:', {traceback.format_exc()})


        
    
