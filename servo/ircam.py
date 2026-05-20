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

from . import faster

log = logging.getLogger(__name__)

class IRCamera(core.Worker):

    def __init__(self, data, events, frame_shape=None, frame_center=None, roi_mode=False):
        
        super().__init__(data, events)

        self.roi_mode = bool(roi_mode)
        log.info(f'IRCamera roi_mode={self.roi_mode}')

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
        if self.roi_mode:
            target_fps = min(max_fps, config.IRCAM_MAX_FPS_ROI_MODE)
        else:
            target_fps = min(max_fps, config.IRCAM_MAX_FPS_FF_MODE)
            
        self.dev.setFps(target_fps)
        self.dev.updateConfig()  #Data is sent to the device
        log.info(f"current fps: {self.dev.fps()}")
        self.data['IRCamera.target_fps'][0] = self.dev.fps()

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

    def onInternalError(self, error_str):
        log.error("onInternalError(" + error_str + ")")


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
        self.arr_roi_normalized = self.data['IRCamera.roi_normalized']
        self.arr_hprofile   = self.data['IRCamera.hprofile']
        self.arr_vprofile   = self.data['IRCamera.vprofile']
        self.arr_hprof_norm = self.data['IRCamera.hprofile_normalized']
        self.arr_vprof_norm = self.data['IRCamera.vprofile_normalized']
        self.arr_hlevels    = self.data['IRCamera.hprofile_levels']
        self.arr_vlevels    = self.data['IRCamera.vprofile_levels']
        self.arr_hlev_pos   = self.data['IRCamera.hprofile_levels_pos']
        self.arr_vlev_pos   = self.data['IRCamera.vprofile_levels_pos']
        self.arr_opds       = self.data['IRCamera.opds']
        self.arr_mean_opd   = self.data['IRCamera.mean_opd']
        self.arr_tip        = self.data['IRCamera.tip']
        self.arr_tilt       = self.data['IRCamera.tilt']
        self.arr_full_output = self.data['IRCamera.full_output']
        
        # Servo coeffs
        self.arr_roinorm_min  = self.data['Servo.roinorm_min']
        self.arr_roinorm_max  = self.data['Servo.roinorm_max']
        
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
        self.hangles  = np.zeros(config.MIN_ROI_SHAPE, dtype=config.DATA_DTYPE)
        self.hlast_angles = np.zeros(config.MIN_ROI_SHAPE, dtype=config.DATA_DTYPE)
        self.vangles  = np.zeros(config.MIN_ROI_SHAPE, dtype=config.DATA_DTYPE)
        self.vlast_angles = np.zeros(config.MIN_ROI_SHAPE, dtype=config.DATA_DTYPE)
        self.hlevels = np.zeros(config.MIN_ROI_SHAPE, dtype=config.DATA_DTYPE)
        self.vlevels = np.zeros(config.MIN_ROI_SHAPE, dtype=config.DATA_DTYPE)
        self.hcenter_pixels_nb = None
        self.vcenter_pixels_nb = None
        

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

        self.last_id = -1
        self.normalization_coeffs_tag = self._get_normalization_coeffs_tag()
        self.roinorm_min = None
        self.roinorm_max = None
        self.roinorm_amp = None
        self.roinorm_mean = None
        self.roinorm_mask = None

        self.last_frame_time = None
        self.target_fps = self.data['IRCamera.target_fps'][0]

        
        gc.freeze()
        gc.disable()


    def _get_normalization_coeffs_tag(self):
        return self.data['Servo.roinorm_min'][:config.TRACKER_TAG_SIZE].copy()
        

    def onNewFrame(self, frame):
        try:                
            frame_id = int(frame.id())
            if frame_id <= self.last_id:
                return
            else:
                self.last_id = frame_id

            # skip frame if loop is too slow i.e. when the time
            # interval between too consecutive frames is two times the
            # target time (to maintain a stable fps)
            frame_time = time.perf_counter()
            if self.last_frame_time is not None:
                if frame_time - self.last_frame_time > 2/self.target_fps:
                    self.last_frame_time = frame_time
                    return
                
            self.last_frame_time = frame_time
                
            index = frame_id - 1
            
            self.times[index % self.times.size] = frame_time
            self.ids[index % self.times.size] = index

            # get stats when buffer fulled
            if index > self.stats_last_index + self.times.size: # buffers were fulled
                self.stats_last_index = (index // self.times.size) * self.times.size               
                mean_sampling_time = float(
                    (np.nanmax(self.times) - np.nanmin(self.times)) / len(self.times))
                self.arr_mean_sampling_time[0] = mean_sampling_time
                self.arr_fps[0] = 1/mean_sampling_time
                self.arr_lost_frames[0] = int(np.sum(np.isnan(self.ids)))
                self.times.fill(np.nan)
                self.ids.fill(np.nan)

            # read frame data as numpy array 
            frame_data = frame.data()  # native order, no transpose

            
            # viewer output + check frame resize
            if ((frame_time - self.last_viewer_out_time > config.IRCAM_VIEWER_OUTPUT_TIME)
                or (self.roinorm_min is None) or (self.roinorm_max is None)):
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
                faster.fast_copy_transpose(frame_data.astype(np.float32, copy=False),
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
                self.iwid = int(self.data['params.PROFILE_WIDTH'][0])

                # reload normalization coefficients if changed (check tag)
                if ((np.any(self._get_normalization_coeffs_tag() != self.normalization_coeffs_tag))
                    or (self.roinorm_min is None)
                    or (self.roinorm_max is None)
                    or (self.roinorm_amp is None)):
                    log.warning('Normalization coefficients changed. Reinitializing IR image buffers.')
            
                    self.roinorm_min = np.array(self.data['Servo.roinorm_min'][:self.profile_len**2]).reshape(
                        (self.profile_len, self.profile_len))
                    self.roinorm_max = np.array(self.data['Servo.roinorm_max'][:self.profile_len**2]).reshape(
                        (self.profile_len, self.profile_len))
                    self.roinorm_amp = self.roinorm_max - self.roinorm_min
                    
                    self.normalization_coeffs_tag = self._get_normalization_coeffs_tag()

                
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
            roi = faster.extract_roi_local_f32(
                frame_data.astype(np.float32, copy=False),
                int(self.iy), int(self.ix), int(self.profile_len))

            
            # # --------- T2: normalization + levels + angles/opd ----------
                            
            roi_norm = utils.normalize_roi(roi, self.roinorm_min, self.roinorm_amp)
                
            ix_profile = self.profile_len//2 + int(self.data['IRCamera.profile_h_shift'][0])
            iy_profile = self.profile_len//2 + int(self.data['IRCamera.profile_v_shift'][0])

            mask = self.roinorm_max.astype(np.float32, copy=False) if self.roinorm_max is not None else None
            vprofile_norm, hprofile_norm = faster.compute_profiles_local_f32(
                roi_norm.astype(np.float32, copy=False),
                int(iy_profile), int(ix_profile), int(self.iwid), int(self.profile_len), get_roi=False, mask=mask)
            
            if compute_servo_output: # Refresh pixel lists when states change
                x_states = self.arr_xpix_states
                y_states = self.arr_ypix_states
            
                if self.x_pixels_states is None or not np.array_equal(self.x_pixels_states, x_states):
                    self.x_pixels_states = np.copy(x_states)
                    try:
                        xpixels_list = utils.get_pixels_lists(x_states)
                        xpixels_list_pos = utils.get_mean_pixels_positions(xpixels_list)
                    except Exception as e:
                        log.error(f'error computing x pixels lists: {e}')
                    else:
                        self.xpixels_list = xpixels_list
                        self.xpixels_list_pos = xpixels_list_pos
                    
                    self.arr_hlev_pos[:3] = self.xpixels_list_pos
                    
                if self.y_pixels_states is None or not np.array_equal(self.y_pixels_states, y_states):
                    self.y_pixels_states = np.copy(y_states)
                    try:
                        ypixels_list = utils.get_pixels_lists(y_states)
                        ypixels_list_pos = utils.get_mean_pixels_positions(ypixels_list)
                    except Exception as e:
                        log.error(f'error computing y pixels lists: {e}')
                    else:
                        self.ypixels_list = ypixels_list
                        self.ypixels_list_pos = ypixels_list_pos
                        
                    self.arr_vlev_pos[:3] = self.ypixels_list_pos

                
            # now that normalized profiles are computed we can compute levels,
            # using the potentially updated pixel lists
            self.hcenter_pixels_nb = len(self.xpixels_list[1])
            self.vcenter_pixels_nb = len(self.ypixels_list[1])

            #faster.batch_compute_levels_f32(
            # mean side, mean center, center_levels
            self.hlevels[:self.hcenter_pixels_nb+2] = utils.compute_profile_levels(
                hprofile_norm, self.xpixels_list)
            
            self.vlevels[:self.vcenter_pixels_nb+2] = utils.compute_profile_levels(
                vprofile_norm, self.ypixels_list)
            
                        
            # Angles & OPD
            # TODO: angles must be computed on ellipse normalized levels #############
            ##########################################################################            
            self.hangles[:self.hcenter_pixels_nb+1] = utils.compute_angles(
                self.hlevels[:self.hcenter_pixels_nb+2], self.arr_hellipse, self.hlast_angles[:self.hcenter_pixels_nb+1])
            self.vangles[:self.vcenter_pixels_nb+1] = utils.compute_angles(
                self.vlevels[:self.vcenter_pixels_nb+2], self.arr_vellipse, self.vlast_angles[:self.vcenter_pixels_nb+1])
            
            hopd = utils.compute_opds(self.hangles[0])
            vopd = utils.compute_opds(self.vangles[0])
            opds = np.array((hopd, vopd), dtype=config.DATA_DTYPE)
            mean_opd = (hopd + vopd) / 2
            mean_opd -= self.arr_mean_opd_offset[0]

            # check for lost
            if self.last_mean_opd is not None and np.isfinite(mean_opd):
                if abs(mean_opd - self.last_mean_opd) > config.IRCAM_LOST_THRESHOLD:
                    log.warning(f'potential lost detected: mean OPD jump from {self.last_mean_opd:.2f} nm to {mean_opd:.2f} nm')
                    self.data['Servo.is_lost'][0] = float(True)

            if np.isfinite(mean_opd):
                self.last_mean_opd = float(mean_opd)                
            
            if compute_servo_output:
                # Normalized profiles (Numba kernels)
                self.arr_hprof_norm[:self.profile_len] = hprofile_norm
                self.arr_vprof_norm[:self.profile_len] = vprofile_norm

                vprofile, hprofile = faster.compute_profiles_local_f32(
                    roi.astype(np.float32, copy=False),
                    int(iy_profile), int(ix_profile), int(self.iwid),
                    int(self.profile_len), get_roi=False, mask=mask)
                
                # Profiles & ROI to shared memory (no .flatten(), roi is contiguous)
                self.arr_hprofile[:self.profile_len] = hprofile
                self.arr_vprofile[:self.profile_len] = vprofile
                self.arr_roi[:roi.size] = roi.ravel()
                self.arr_roi_normalized[:roi_norm.size] = roi_norm.ravel()

                # Levels & OPDs
                self.arr_hlevels[:2] = self.hlevels[:2]
                self.arr_vlevels[:2] = self.vlevels[:2]
                
                self.arr_opds[:2] = opds.astype(config.FRAME_DTYPE)
                self.arr_mean_opd[0] = float(mean_opd)

            tip = np.mean(np.diff(self.hangles[1:self.hcenter_pixels_nb+1]))
            tilt = np.mean(np.diff(self.vangles[1:self.vcenter_pixels_nb+1]))
            self.arr_tip[0]  = tip
            self.arr_tilt[0] = tilt
                
                
            self.hlast_angles = self.hangles
            self.vlast_angles = self.vangles
            loop_time = time.perf_counter() - frame_time
            self.arr_loop_time[0] = loop_time
            self.arr_loop_fps[0] = 1./loop_time
            
            
        except Exception as e:
            log.error(f'error on new frame: {e}, traceback: {traceback.format_exc()}')
            
            

        
    
