import numpy as np
import time
import logging
import traceback

from . import NITLibrary_x64_382_py312 as NITLibrary

from . import core
from . import config
from . import utils

log = logging.getLogger(__name__)

class IRCamera(core.Worker):

    def __init__(self, data, events, roi_shape=None, roi_center=None):
        
        super().__init__(data, events)
        
        if roi_shape is None:
            self.roi_shape = np.array(config.FULL_FRAME_SHAPE)
            
        if roi_center is None:
            self.roi_position = np.array(config.DEFAULT_ROI_POSITION)
        else:
            self.roi_position = np.array(roi_center) - roi_shape//2
            
        assert (0 <= self.roi_position[0] < config.FULL_FRAME_SHAPE[0]), f'bad x position: {self.roi_position[0]}'
        assert (0 <= self.roi_position[1] < config.FULL_FRAME_SHAPE[1]), f'bad y position: {self.roi_position[1]}'

        assert (config.MIN_ROI_SHAPE <= self.roi_shape[0] <= config.FULL_FRAME_SHAPE[0]), f'bad shape: {self.roi_shape[0]}'
        assert (config.MIN_ROI_SHAPE <= self.roi_shape[1] <= config.FULL_FRAME_SHAPE[1]), f'bad shape: {self.roi_shape[1]}'

        log.info(f'target ROI {self.roi_shape} at {self.roi_position}')

        self.roi_size = np.prod(self.roi_shape)
        
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
        
        self.dev.setParamValueOf("Number Of Columns", int(self.roi_shape[0]))
        self.dev.setParamValueOf("Number Of Lines", int(self.roi_shape[1]))
        self.dev.setParamValueOf("First Column", int(self.roi_position[0]))
        self.dev.setParamValueOf("First Line", int(self.roi_position[1]))
        
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
        self.data_observer = DataObserver(self.data, self.roi_position, self.roi_shape)
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

    
    def __init__(self, data, roi_position, roi_shape):

        super().__init__()
        self.data = data
        self.roi_position = roi_position
        self.roi_shape = roi_shape
        self.times = np.full(1000, np.nan, dtype=float)
        self.ids = np.full_like(self.times, np.nan)
        self.last_frame_out_time = 0
        self.frame_out_period = 0.1 # s
        self.stats_last_index = 0
        self.x_pixels_states = None
        self.y_pixels_states = None
        self.xpixels_list = None
        self.ypixels_list = None
        self.xpixels_list_pos = None
        self.ypixels_list_pos = None
        
        
    def onNewFrame(self, frame, get_roi=True):
        try:
            index = int(frame.id()) - 1
            frame_time = time.time()
            self.times[index % self.times.size] = frame_time
            self.ids[index % self.times.size] = index

            if index > self.stats_last_index + self.times.size: # buffers were fulled
                self.stats_last_index = (index // self.times.size) * self.times.size
                diff_times = np.diff(self.times)
                diff_ids = np.diff(self.ids)
                log.info(f'frame {index} median sampling time: {np.nanmedian(diff_times)}, {np.sum(diff_times==0)}, {np.sum(diff_times==0)}, {np.sum(diff_ids!=1)}')
                self.times.fill(np.nan)
                self.ids.fill(np.nan)
            
            if frame_time - self.last_frame_out_time > self.frame_out_period:
                self.last_frame_out_time = frame_time

                frame_data = frame.data().T.flatten()
                self.data['IRCamera.last_frame'][:self.data['IRCamera.frame_size'][0]] = frame_data

            try:
                # profile x and y always set in full frame coordinates
                ix = self.data['IRCamera.profile_x'][0] - self.roi_position[0]
                iy = self.data['IRCamera.profile_y'][0] - self.roi_position[1]
                profile_len = self.data['IRCamera.profile_len'][0]
                profile_width = self.data['IRCamera.profile_width'][0]
                ilen = int(profile_len)
                if ilen > np.min(self.roi_shape): ilen = np.min(self.roi_shape)
                iwid = int(profile_width)

                hprofile, vprofile, roi = utils.compute_profiles(frame.data().T, ix, iy,
                                                                 iwid, ilen,
                                                                 get_roi=get_roi)

                # nan padded versions
                if hprofile.size != ilen:
                    hprofile_np = np.full(profile_len, np.nan, dtype=hprofile.dtype)
                    hprofile_np[:np.size(hprofile)] = hprofile
                else:
                    hprofile_np = hprofile

                if vprofile.size != ilen:
                    vprofile_np = np.full(profile_len, np.nan, dtype=vprofile.dtype)
                    vprofile_np[:np.size(vprofile)] = vprofile
                else:
                    vprofile_np = vprofile


                self.data['IRCamera.hprofile'][:profile_len] = hprofile_np
                self.data['IRCamera.vprofile'][:profile_len] = vprofile_np
                if get_roi:
                    self.data['IRCamera.roi'][:profile_len**2] = roi.flatten()

                # normalized profiles
                hnorm_min = self.data['Servo.hnorm_min'][:profile_len]
                hnorm_max = self.data['Servo.hnorm_max'][:profile_len]
                vnorm_min = self.data['Servo.vnorm_min'][:profile_len]
                vnorm_max = self.data['Servo.vnorm_max'][:profile_len]
                hprofile_normalized = utils.normalize_profile(
                    hprofile_np, hnorm_min, hnorm_max)
                vprofile_normalized = utils.normalize_profile(
                    vprofile_np, vnorm_min, vnorm_max)
                self.data['IRCamera.hprofile_normalized'][:profile_len] = hprofile_normalized
                self.data['IRCamera.vprofile_normalized'][:profile_len] = vprofile_normalized
                
                # compute levels
                
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
                    
                hlevels = utils.compute_profile_levels(
                    hprofile_normalized, self.xpixels_list)
                vlevels = utils.compute_profile_levels(
                    vprofile_normalized, self.ypixels_list)

                self.data['IRCamera.hprofile_levels'][:3] = hlevels
                self.data['IRCamera.vprofile_levels'][:3] = vlevels
                
                # compute angles and opd
                last_angles = self.data['IRCamera.last_angles'][:4]
                opds = np.empty_like(last_angles)
                
                hellipse_norm_coeffs = self.data['Servo.hellipse_norm_coeffs'][:4]
                vellipse_norm_coeffs = self.data['Servo.vellipse_norm_coeffs'][:4]
                
                hangles = utils.compute_angles(hlevels, hellipse_norm_coeffs)
                hangles = utils.unwrap_angles(hangles, last_angles[:2])
                hopds = utils.compute_opds(hangles)
                
                vangles = utils.compute_angles(vlevels, vellipse_norm_coeffs)
                vangles = utils.unwrap_angles(vangles, last_angles[2:])
                vopds = utils.compute_opds(vangles)

                opds[:2] = hopds
                opds[2:] = vopds
                self.data['IRCamera.opds'][:4] = opds.astype(config.FRAME_DTYPE)
                
                last_angles[:2] = hangles
                last_angles[2:] = vangles
                self.data['IRCamera.last_angles'][:4] = last_angles.astype(config.FRAME_DTYPE)
                
                
                
                        
            except Exception as e:
                log.error(f'Error at reading profiles on camera {traceback.format_exc()}')
        
        except Exception as e:
            log.error('error on new frame:', {traceback.format_exc()})


        
    
