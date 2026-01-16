import numpy as np
import logging
from multiprocessing import shared_memory

from . import config
from . import state_store


log = logging.getLogger(__name__)


class Worker(object):

    def __init__(self, data, stop_event):
        self.data = data
        self.stop_event = stop_event
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

        init_levels = np.array(self.state.get("DAQ.piezos_level"),
                               dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
        if np.any(np.isnan(init_levels)):
            init_levels = np.zeros(3, dtype=config.DAQ_PIEZO_LEVELS_DTYPE)
            log.warning('piezos levels could not be loaded from saved state')
        else:
            log.info(f'init piezos levels: {init_levels}')
            
        self.add_array('DAQ.piezos_level', init_levels)

    
    def add_array(self, name, array):
        try:
            self.shms[name] = shared_memory.SharedMemory(create=True, name=name, size=array.nbytes)
        except FileExistsError:
            self.shms[name] = shared_memory.SharedMemory(name=name)
            
        self.arrs[name] = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shms[name].buf)
        
        # Remplissage vectorisé (rapide, côté C) :
        self.arrs[name][:] = array
        #self.shms[name].close()  # ne détruit pas
        log.info(f'added shared array {name} {array.shape}')

    def add_value(self, name, val):
        self.add_array(name, np.array([val,]))


    def __getitem__(self, name):
        return self.arrs[name]

    def __setitem__(self, name, value):
        self.arrs[name] = value

    def stop(self):
        
        # save states
        log.info('saving states')
        self.state.set("DAQ.piezos_level", list(self["DAQ.piezos_level"][:3]))

        self.state.save()

        # free shared memory
        for ikey in self.shms:
            self.shms[ikey].close()
            self.shms[ikey].unlink()
            log.info(f'removed shared memory {ikey}')




