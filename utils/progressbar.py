import sys
import os
import time
import IPython


class ProgressBar(object):
    def __init__(self, bar_width=None, title='', initial_progress=0, end_value=None, done_symbol='#', wait_symbol='-'):
        title += ": " if title != '' else ''
        self.title = title
        self._c = 7  # Lenth of the "[] xxx%" part of printed string
        if bar_width is None:
            try:
                _, bar_width = os.popen('stty size', 'r').read().split()
            except Exception:
                bar_width = 10 + len(title + self._c)
        self._w = int(bar_width) - len(self.title) - self._c  # Subtract constant parts of string length
        assert self._w >= 0, 'Title too long, bar width too narrow or terminal window not wide enough'
        self._b = self._w + len(self.title) + self._c  # Number of left shifts to apply at end to reset to head of line
        self.end_value = end_value
        self.ds = done_symbol
        self.ws = wait_symbol
        self.initial_x = initial_progress

    def start(self):
        """Creates a progress bar `width` chars long on the console
        and moves cursor back to beginning with BS character"""
        self.progress(self.initial_x)

    def progress(self, x):
        """Sets progress bar to a certain percentage x if `end_value`
        is `None`, otherwise, computes `x` as percentage of `end_value`."""
        assert x <= 1 or self.end_value is not None and self.end_value >= x
        if self.end_value is not None:
            x = x / self.end_value
        y = int(x * self._w)                      
        sys.stdout.write(self.title + "[" + self.ds * y + self.ws * (self._w - y) + "] {:3d}%".format(int(round(x * 100))) + chr(8) * (self._w + len(self.title) + self._c))
        sys.stdout.flush()

    def end(self):
        """End of progress bar.
        Write full bar, then move to next line"""
        sys.stdout.write(self.title + "[" + self.ds * self._w + "] {:3d}%".format(100) + "\n")
        sys.stdout.flush()


def startprogress(title):
    """Creates a progress bar 40 chars long on the console
    and moves cursor back to beginning with BS character"""
    global progress_x
    sys.stdout.write(title + ": [" + "-" * 40 + "]" + chr(8) * 41)
    sys.stdout.flush()
    progress_x = 0


def progress(x):
    """Sets progress bar to a certain percentage x.
    Progress is given as whole percentage, i.e. 50% done
    is given by x = 50"""
    global progress_x
    x = int(x * 40 // 100)                      
    sys.stdout.write("#" * x + "-" * (40 - x) + "]" + chr(8) * 41)
    sys.stdout.flush()
    progress_x = x


def endprogress():
    """End of progress bar.
    Write full bar, then move to next line"""
    sys.stdout.write("#" * 40 + "]\n")
    sys.stdout.flush()


def progressBar(value, endvalue, bar_length=20):
    """Write and update a progress bar.
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()

# Testing
if __name__ == '__main__':
    sleep = 0.12
    pb = ProgressBar(title='Test: 20s with 4s elapsed and some other stuff', initial_progress=4, end_value=100)
    pb.start()
    time.sleep(sleep)
    for i in range(5, 100):
        pb.progress(i)
        time.sleep(sleep)
    pb.end()

    IPython.embed()
    