#-*- coding: utf-8 -*-

import time

__all__ = ['Timer']


class Timer(object):
    def __init__(self):
        self.clear()

    def start(self):
        assert self._timing == False
        self._timing = True
        self._start_time = time.time()

    def stop(self):
        assert self._timing == True
        self._record += time.time() - self._start_time
        self._timing = False

    def clear(self):
        self._start_time = 0.
        self._record = 0.
        self._timing = False

    def get(self):
        return self._record

    def pop(self):
        ret = self._record
        self.clear()
        return ret