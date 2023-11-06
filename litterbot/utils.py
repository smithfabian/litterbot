import cv2
from traitlets import HasTraits, Float, observe


def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])



class Sliders(HasTraits):
    
    def __init__(self, **kwargs):
        super().__init__()
        
        for key, obj in kwargs.items():
            if isinstance(obj, HasTraits) and hasattr(obj, 'value'):
                setattr(self, key, obj.value)
                
                def _observer_callback(change, name=key):
                    setattr(self, name, change.new)
                
                obj.observe(_observer_callback, names='value')