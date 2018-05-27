import re
from collections import defaultdict
from enum import Enum


# TODO not to allow a name of variable started from '_'

class SEMOperations(Enum):
    REGRESSION = '~'
    MEASUREMENT = '=~'
    COVARIANCE = '~~'
    TYPE = ':'


class SEMParser:
    s_pattern = r'\b(\S+(?:\s*\+\s*\S+)*)\s*({})\s*(\S+(?:\s*(?:\+|\*)\s*\S+)*)\b'

    def __init__(self):
        self.m_opDict = {}
        self.operations = SEMOperations
        s = []
        for op in self.operations:
            self.m_opDict[op.value] = op
            s.append(op.value)
        self.m_pattern = self.s_pattern.format('|'.join(s))

    def parse(self, string):
        # Ignore comments.
        strings = [v.split('#')[0] for v in string.splitlines()]
        d = defaultdict(lambda: {op: defaultdict(lambda: []) for op in self.operations})
        for s in strings:
            r = re.search(self.m_pattern, s)
            if r:
                lvalues = [val.strip() for val in r.group(1).split('+')]
                rvalues = [val.strip().split('*')[::-1]
                           for val in r.group(3).split('+')]
                op = self.m_opDict[r.group(2)]
                for lvalue in lvalues:
                    for rvalue in rvalues:
                        d[rvalue[0]]
                        d[lvalue][op][rvalue[0]] += rvalue[1:]

        if 'binary' in d.keys():
            d.pop('binary')
        if 'ordinal' in d.keys():
            d.pop('ordinal')
        return d
