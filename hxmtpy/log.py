from __future__ import absolute_import, division
import numpy as np
import numba 
from functools import wraps

class Log():

    def log_paras(func):
        '''
        function decorator to record the parameters used in function
        '''
        @wraps(func)    
        def log_wrapper(*args, **kwargs):
            results = func(*args, **kwargs)
            if ('history' in kwargs) and (kwargs['history']):
                log_paras_dict = kwargs
                del log_paras_dict['history']
                return results, log_paras_dict
            else:
                return results
        return log_wrapper

    def log_files(keyword, value):
        return {keyword: value}

