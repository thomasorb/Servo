import numpy as np
import time
import logging
import traceback
import collections
import gc

from . import NITLibrary_x64_382_py312 as NITLibrary

from . import core
from . import config
from . import utils

from .faster import fast_copy_transpose, compute_profiles_local_f32, batch_normalize_and_levels

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
        gc.enable()
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
        self.last_mean_opd = None
        
        # Cache shared buffers for fast access (1 lookup per name)
        self.arr_last_frame = self.data['IRCamera.last_frame']
        self.arr_roi        = self.data['IRCamera.roi']
        self.arr_hprofile   = self.data['IRCamera.hprofile']
        self.arr_vprofile   = self.data['IRCamera.vprofile']
        self.arr_hprof_norm = self.data['IRCamera.hprofile_normalized']
        self.arr_vprof_norm = self.data['IRCamera.vprofile_normalized']
        self.arr_hlevels    = self.data['IRCamera.hprofile_levels']
        self.arr_vlevels    = self.data['IRCamera.vprofile_levels']
        self.arr_hlev_pos   = self.data['IRCamera.hprofile_levels_pos']
        self.arr_vlev_pos   = self.data['IRCamera.vprofile_levels_pos']
        self.arr_last_angles= self.data['IRCamera.last_angles']
        self.arr_opds       = self.data['IRCamera.opds']
        self.arr_mean_opd   = self.data['IRCamera.mean_opd']
        self.arr_std_opd    = self.data['IRCamera.std_opd']
        self.arr_mean_opd_buf = self.data['IRCamera.mean_opd_buffer']
        self.arr_tip        = self.data['IRCamera.tip']
        self.arr_tilt       = self.data['IRCamera.tilt']
        self.arr_tip_buf    = self.data['IRCamera.tip_buffer']
        self.arr_tilt_buf   = self.data['IRCamera.tilt_buffer']
        self.arr_time_buf       = self.data['IRCamera.time_buffer']
        self.arr_velocity   = self.data['IRCamera.velocity']
        self.arr_velocity_buf   = self.data['IRCamera.velocity_buffer']
        self.arr_full_output = self.data['IRCamera.full_output']
        
        # Servo coeffs
        self.arr_hnorm_min  = self.data['Servo.hnorm_min']
        self.arr_hnorm_max  = self.data['Servo.hnorm_max']
        self.arr_vnorm_min  = self.data['Servo.vnorm_min']
        self.arr_vnorm_max  = self.data['Servo.vnorm_max']
        self.arr_xpix_states= self.data['Servo.pixels_x']
        self.arr_ypix_states= self.data['Servo.pixels_y']
        self.arr_hellipse   = self.data['Servo.hellipse_norm_coeffs']
        self.arr_vellipse   = self.data['Servo.vellipse_norm_coeffs']

        # values
        self.arr_mean_sampling_time = self.data['IRCamera.mean_sampling_time']
        self.arr_fps = self.data['IRCamera.fps']
        self.arr_lost_frames = self.data['IRCamera.lost_frames']
        self.arr_loop_time = self.data['IRCamera.loop_time']
        self.arr_loop_fps = self.data['IRCamera.loop_fps']
        self.arr_frame_dimx = self.data['IRCamera.frame_dimx']
        self.arr_frame_dimy = self.data['IRCamera.frame_dimy']
        self.arr_frame_size = self.data['IRCamera.frame_size']
        self.arr_mean_opd_offset = self.data['IRCamera.mean_opd_offset']

        # Preallocated workspaces to avoid per-frame allocations
        self.hlevels_ws = np.empty(3, dtype=config.DATA_DTYPE)
        self.vlevels_ws = np.empty(3, dtype=config.DATA_DTYPE)
        self.angles_ws  = np.empty(4, dtype=config.DATA_DTYPE)

        # Pixel lists cache
        self.x_pixels_states = None
        self.y_pixels_states = None
        self.xpixels_list = None
        self.ypixels_list = None
        self.xpixels_list_pos = None
        self.ypixels_list_pos = None

        self.ix = None
        self.iy = None
        self.profile_len = None
        self.iwid = None

        self.opd_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.tip_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.tilt_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.time_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.velocity_deque = collections.deque(maxlen=config.SERVO_BUFFER_SIZE)
        self.last_id = -1

        gc.freeze()
        gc.disable()

    def onNewFrame(self, frame):
        try:
            frame_time = time.perf_counter()
            frame_id = int(frame.id())
            if frame_id == self.last_id:
                return
            else:
                self.last_id = frame_id
                
            index = frame_id - 1
            
            self.times[index % self.times.size] = frame_time
            self.ids[index % self.times.size] = index

            # get stats when buffer fulled
            if index > self.stats_last_index + self.times.size: # buffers were fulled
                self.stats_last_index = (index // self.times.size) * self.times.size               
                diff_ids = np.diff(self.ids)
                mean_sampling_time = float(
                    (np.nanmax(self.times) - np.nanmin(self.times)) / np.sum(
                        ~np.isnan(self.times)))
                self.arr_mean_sampling_time[0] = mean_sampling_time
                self.arr_fps[0] = 1/mean_sampling_time
                self.arr_lost_frames[0] = int(np.sum(diff_ids > 1))
                self.times.fill(np.nan)
                self.ids.fill(np.nan)

            frame_data = frame.data()  # native order, no transpose
    
            # --------- T0: viewer copy (and potential resize) ----------                
            if frame_time - self.last_viewer_out_time > config.IRCAM_VIEWER_OUTPUT_TIME:
                self.last_viewer_out_time = frame_time

                # Ensure last_frame has correct size 
                expected_size = frame_data.shape[0] * frame_data.shape[1]
                if expected_size != self.data['IRCamera.frame_size'][0]:
                    # Update metadata
                    self.arr_frame_dimx[0] = int(frame_data.shape[0])
                    self.arr_frame_dimy[0] = int(frame_data.shape[1])
                    self.arr_frame_size[0] = int(expected_size)

                # Copy transposed view directly (no flatten temp)
                # Keep the transpose to preserve viewer orientation
                fast_copy_transpose(frame_data.astype(np.float32, copy=False),
                    self.arr_last_frame[:expected_size])

                # Determine profile geometry
                if self.roi_mode:
                    self.ix, self.iy = self.frame_shape // 2
                    self.profile_len = min(self.frame_shape)
                else:
                    self.ix = self.data['IRCamera.profile_x'][0] - self.frame_position[0]
                    self.iy = self.data['IRCamera.profile_y'][0] - self.frame_position[1]
                    self.profile_len = int(self.data['IRCamera.profile_len'][0])
                if self.profile_len > np.min(self.frame_shape):
                    self.profile_len = int(np.min(self.frame_shape))
                self.iwid = int(self.data['IRCamera.profile_width'][0])

                
            if bool(self.arr_full_output[0]):
                compute_servo_output = True
            else:
                compute_servo_output = (frame_time - self.last_servo_out_time > config.IRCAM_SERVO_OUTPUT_TIME)

            if compute_servo_output:
                self.last_servo_out_time = frame_time


            # Compute profiles + ROI (Numba accelerated)

            # When the code previously used `a.T`, (ix, iy) in that transposed space
            # correspond to (iy, ix) in the native space.
            #vprofile, hprofile, roi = utils.compute_profiles(
            #    frame_data, self.iy, self.ix, self.iwid, self.profile_len, get_roi=True)
            vprofile, hprofile, roi = compute_profiles_local_f32(
                frame_data.astype(np.float32, copy=False),
                int(self.iy), int(self.ix), int(self.iwid), int(self.profile_len), True)

            
            # # --------- T2: normalization + levels + angles/opd ----------
                            
            self.hmin = self.arr_hnorm_min[:self.profile_len]
            self.hmax = self.arr_hnorm_max[:self.profile_len]
            self.vmin = self.arr_vnorm_min[:self.profile_len]
            self.vmax = self.arr_vnorm_max[:self.profile_len]

            if compute_servo_output:

                # Refresh pixel lists when states change
                x_states = self.arr_xpix_states
                y_states = self.arr_ypix_states
                if self.x_pixels_states is None or not np.array_equal(self.x_pixels_states, x_states):
                    self.x_pixels_states = np.copy(x_states)
                    self.xpixels_list = utils.get_pixels_lists(x_states)
                    self.xpixels_list_pos = utils.get_mean_pixels_positions(self.xpixels_list)
                    self.arr_hlev_pos[:3] = self.xpixels_list_pos
                if self.y_pixels_states is None or not np.array_equal(self.y_pixels_states, y_states):
                    self.y_pixels_states = np.copy(y_states)
                    self.ypixels_list = utils.get_pixels_lists(y_states)
                    self.ypixels_list_pos = utils.get_mean_pixels_positions(self.ypixels_list)
                    self.arr_vlev_pos[:3] = self.ypixels_list_pos

                # Normalized profiles (Numba kernels)
                self.arr_hprof_norm[:self.profile_len] = utils.normalize_profile(
                    hprofile, self.hmin, self.hmax, inplace=False)
                self.arr_vprof_norm[:self.profile_len] = utils.normalize_profile(
                    vprofile, self.vmin, self.vmax, inplace=False)

                
            # Levels (3 positions per axis)
            self.hlevels_ws[:] = batch_normalize_and_levels(
                hprofile, self.hmin, self.hmax,
                self.xpixels_list[0], self.xpixels_list[1], self.xpixels_list[2]
            )
            self.vlevels_ws[:] = batch_normalize_and_levels(
                vprofile, self.vmin, self.vmax,
                self.ypixels_list[0], self.ypixels_list[1], self.ypixels_list[2]
            )
            
            # Angles & OPD
            self.angles_ws[:2] = utils.compute_angles(self.hlevels_ws, self.arr_hellipse[:4],
                                                      self.arr_last_angles[:2])
            self.angles_ws[2:] = utils.compute_angles(self.vlevels_ws, self.arr_vellipse[:4],
                                                      self.arr_last_angles[2:])
            opds = utils.compute_opds(self.angles_ws)
            opds -= self.arr_mean_opd_offset[0]
            mean_opd = utils.mean(opds)

            # check for lost
            if self.last_mean_opd is not None:
                if abs(mean_opd - self.last_mean_opd) > config.IRCAM_LOST_THRESHOLD:
                    log.warning(f'potential lost detected: mean OPD jump from {self.last_mean_opd:.2f} nm to {mean_opd:.2f} nm')
                    self.data['Servo.is_lost'][0] = float(True)
            self.last_mean_opd = float(mean_opd)
                        
            
            if compute_servo_output:
                
                # Profiles & ROI to shared memory (no .flatten(), roi is contiguous)
                self.arr_hprofile[:self.profile_len] = hprofile
                self.arr_vprofile[:self.profile_len] = vprofile
                self.arr_roi[:roi.size] = roi.ravel()

                # Levels & OPDs
                self.arr_hlevels[:3] = self.hlevels_ws
                self.arr_vlevels[:3] = self.vlevels_ws
                self.arr_opds[:4] = opds.astype(config.FRAME_DTYPE)
                self.arr_mean_opd[0] = float(mean_opd)

                tip  = self.angles_ws[1] - self.angles_ws[0]
                tilt = self.angles_ws[3] - self.angles_ws[2]
                self.arr_tip[0]  = tip
                self.arr_tilt[0] = tilt

                self.time_deque.appendleft(float(time.perf_counter()))
                self.arr_time_buf[:min(
                    len(self.time_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                        self.time_deque, dtype=config.FRAME_DTYPE)
                
                self.opd_deque.appendleft(float(mean_opd))
                self.arr_mean_opd_buf[:min(
                    len(self.opd_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                        self.opd_deque, dtype=config.FRAME_DTYPE)
                
                self.arr_std_opd[0] = float(np.std(self.opd_deque))
                
                self.tip_deque.appendleft(float(tip))
                self.arr_tip_buf[:min(
                    len(self.tip_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                        self.tip_deque, dtype=config.FRAME_DTYPE)
                 
                self.tilt_deque.appendleft(float(tilt))
                self.arr_tilt_buf[:min(
                    len(self.tilt_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                        self.tilt_deque, dtype=config.FRAME_DTYPE)

                if len(self.time_deque) > 1:
                    velocity = ((self.opd_deque[0] - self.opd_deque[1])
                                / (self.time_deque[0] - self.time_deque[1]))
                else:
                    velocity = np.nan
                self.arr_velocity[0] = float(velocity)
                self.velocity_deque.appendleft(float(velocity))
                self.arr_velocity_buf[:min(
                    len(self.velocity_deque), config.SERVO_BUFFER_SIZE)] = np.array(
                        self.velocity_deque, dtype=config.FRAME_DTYPE)

                
            self.arr_last_angles[:4] = self.angles_ws
            loop_time = time.perf_counter() - frame_time
            self.arr_loop_time[0] = loop_time
            self.arr_loop_fps[0] = 1./loop_time
            
            
        except Exception as e:
            log.error(f'error on new frame: {e}')

        
    
