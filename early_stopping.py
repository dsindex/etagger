# source from 
# : http://forensics.tistory.com/29

from __future__ import print_function

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience  = patience
        self.verbose = verbose
 
    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('Training process is stopped early!')
                return True
        else:
            self._step = 0
            self._loss = loss
 
        return False
