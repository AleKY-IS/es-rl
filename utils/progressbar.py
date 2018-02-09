import sys
import os


class ProgressBar(object):
    def __init__(self, bar_width=None, title='', initial_progress=0, end_value=None, done_symbol='#', wait_symbol='-'):
        if bar_width is None:
            _, bar_width = os.popen('stty size', 'r').read().split()
        title += ": " if title != '' else ''
        self.w = int(bar_width) - len(title) - 7  # Subtract title length and the "[] xxx%" part of printed string
        self.title = title
        self.end_value = end_value
        self.sd = done_symbol
        self.sw = wait_symbol
        self.start_string = self.title + "[" + self.sd * initial_progress + self.sw * (self.w - initial_progress) + "]" + chr(8) * (self.w + 1)

    def startprogress(self):
        """Creates a progress bar `width` chars long on the console
        and moves cursor back to beginning with BS character"""
        sys.stdout.write(self.start_string)
        sys.stdout.flush()

    def progress(self, x):
        """Sets progress bar to a certain percentage x if `end_value`
        is `None`, otherwise, computes `x` as percentage of `end_value`."""
        assert x <= 1 or self.end_value is not None and self.end_value >= x
        if self.end_value is not None:
            x = x / self.end_value
        y = int(x * self.w)                      
        sys.stdout.write(self.title + "[" + self.sd * y + self.sw * (self.w - y) + "] {:3d}%".format(int(round(x * 100))) + chr(8) * (self.w + 1))
        sys.stdout.flush()

    def endprogress(self):
        """End of progress bar.
        Write full bar, then move to next line"""
        sys.stdout.write(self.title + "[" + self.sd * self.w + "] {:3d}%".format(100) + "\n")
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
