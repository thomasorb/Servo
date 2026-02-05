import numpy as np
import logging
from multiprocessing import shared_memory

from . import config
from . import state_store


log = logging.getLogger(__name__)


class Worker(object):

    def __init__(self, data, events):
        self.data = data
        self.events = events
        self.stop_event = self.events[f"{self.__class__.__name__}.stop"]
        self.running = True

    
    def run(self):
        """Main worker loop."""
        while not self.stop_event.is_set() and self.running:
            self.loop_once()
        
    def cleanup(self):
        """Optional: close hardware / files / buffers."""
        pass

    def stop(self):
        """Always called when shutting down."""
        self.running = False
        self.cleanup()



class SharedData(object):

    def __init__(self):
        self.shms = dict()
        self.arrs = dict()
        self.stored_names = list()

        self.state = state_store.StateStore(
            app_name="servo",
            filename="state.json",
            schema_version=1,
            autosave_interval=10.0,   # save every 10 seconds
            only_main_process_writes=True
        )
        
        self.add_array('IRCamera.last_frame', np.zeros(config.FULL_FRAME_SIZE, dtype=config.FRAME_DTYPE))
        self.add_value('IRCamera.frame_size', int(config.FULL_FRAME_SIZE))
        self.add_value('IRCamera.frame_dimx', int(config.FULL_FRAME_SHAPE[0]))
        self.add_value('IRCamera.frame_dimy', int(config.FULL_FRAME_SHAPE[1]))
        self.add_value('IRCamera.initialized', False)

        # profiles len set to max to avoid problems when changing profile length
        self.add_array('IRCamera.hprofile', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                     dtype=config.FRAME_DTYPE)) 
        self.add_array('IRCamera.vprofile', np.zeros(config.FULL_FRAME_SHAPE[1],
                                                     dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.hprofile_normalized', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                                dtype=config.FRAME_DTYPE)) 
        self.add_array('IRCamera.vprofile_normalized', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                                dtype=config.FRAME_DTYPE)) 
        
        
        self.add_array('IRCamera.hprofile_levels', np.zeros(3, dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.vprofile_levels', np.zeros(3, dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.hprofile_levels_pos', np.zeros(3, dtype=config.FRAME_DTYPE))
        self.add_array('IRCamera.vprofile_levels_pos', np.zeros(3, dtype=config.FRAME_DTYPE))
        
        
        self.add_value('IRCamera.profile_x', int(config.DEFAULT_PROFILE_POSITION[0]), stored=True)
        self.add_value('IRCamera.profile_y', int(config.DEFAULT_PROFILE_POSITION[1]), stored=True)
        self.add_value('IRCamera.profile_len', int(config.DEFAULT_PROFILE_LEN), stored=True)
        self.add_value('IRCamera.profile_width', int(config.DEFAULT_PROFILE_WIDTH), stored=True)
        self.add_array('IRCamera.roi', np.zeros(config.FULL_FRAME_SIZE, dtype=config.FRAME_DTYPE))

        self.add_array('IRCamera.angles', np.zeros(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('IRCamera.last_angles', np.zeros(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('IRCamera.opds', np.zeros(4, dtype=config.FRAME_DTYPE), stored=True)
        

        # selected pixels: 0:none, 1:side, 2:center
        self.add_array('Servo.pixels_x', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                  dtype=int), stored=True)
        self.add_array('Servo.pixels_y', np.zeros(config.FULL_FRAME_SHAPE[1],
                                                  dtype=int), stored=True)
        
        
        self.add_array('Servo.roinorm_min', np.zeros(config.FULL_FRAME_SIZE,
                                                     dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.roinorm_max', np.ones(config.FULL_FRAME_SIZE,
                                                    dtype=config.FRAME_DTYPE), stored=True)

        self.add_array('Servo.hnorm_min', np.zeros(config.FULL_FRAME_SHAPE[0],
                                                      dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.hnorm_max', np.ones(config.FULL_FRAME_SHAPE[0],
                                                     dtype=config.FRAME_DTYPE), stored=True) 
        self.add_array('Servo.vnorm_min', np.zeros(config.FULL_FRAME_SHAPE[1],
                                                      dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.vnorm_max', np.ones(config.FULL_FRAME_SHAPE[1],
                                                     dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.hellipse_norm_coeffs',
                       np.ones(4, dtype=config.FRAME_DTYPE), stored=True)
        self.add_array('Servo.vellipse_norm_coeffs',
                       np.ones(4, dtype=config.FRAME_DTYPE), stored=True)

        

        
        self.add_array('DAQ.piezos_level',
                       np.zeros(3, dtype=config.DAQ_PIEZO_LEVELS_DTYPE),
                       stored=True)
        
        self.add_array('DAQ.piezos_level_actual',
                       np.zeros(3, dtype=config.DAQ_PIEZO_LEVELS_DTYPE),
                       stored=True)


    def add_array(self, name, array, stored=False):
        if stored:
            self.stored_names.append(name)
            if self.state.get(name) is None:
                log.warning(f'{name} could not be loaded from saved states')
            else:
                init = np.array(self.state.get(name), dtype=array.dtype)
                if np.any(np.isnan(init)):
                    log.warning(f'{name} could not be loaded from saved states')
                else:
                    array = init
            
        try:
            self.shms[name] = shared_memory.SharedMemory(create=True, name=name, size=array.nbytes)
        except FileExistsError:
            self.shms[name] = shared_memory.SharedMemory(name=name)
            
        self.arrs[name] = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shms[name].buf)
        
        # Remplissage vectorisé (rapide, côté C) :
        self.arrs[name][:] = array
        #self.shms[name].close()  # ne détruit pas
        log.info(f'added shared array {name} {array.shape}')

    def add_value(self, name, val, stored=False):
        self.add_array(name, np.array([val,]), stored=stored)


    def __getitem__(self, name):
        return self.arrs[name]

    def __setitem__(self, name, value):
        self.arrs[name] = value

    def stop(self):
        
        # save states
        log.info('saving states')
        for iname in self.stored_names:
            idata = list(self[iname][:])
            self.state.set(iname, idata)
            log.info(f'   {iname} : {idata}')
                    
        self.state.save()

        # free shared memory
        for ikey in self.shms:
            self.shms[ikey].close()
            try:
                self.shms[ikey].unlink()
                log.info(f'removed shared memory {ikey}')
            except FileNotFoundError:
                log.warning(f'shared memory {ikey} could not be unlinked')




