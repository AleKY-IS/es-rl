import sys
import numpy as np
import decimal


d = {
    "int": 0,
    "float": 0.0,
    "dict": dict(),
    "set": set(),
    "tuple": tuple(),
    "list": list(),
    "str": "a",
    "unicode": u"a",
    "decimal": decimal.Decimal(0),
    "object": object(),
    "np.int64": np.int64(1),
    "np.int32": np.int32(1),
    "np.float": np.float(1),
    "np.str": np.str('a'),

}

for k, v in sorted(d.items()):
    s = '{0:8s}  {1:d}'.format(k, sys.getsizeof(v))
    print(s)


# value_type = 'int'
# array_likes = ['list', 'set', 'tuple', 'str', 'np.array']
# for array_type in array_likes:
#     for 
#     eval(array_type + '(v)')
#     print()
