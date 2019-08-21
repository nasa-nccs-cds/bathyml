from typing import List, Optional, Tuple, Dict, Any


class ParameterSpace:

    def __init__(self, spec: Dict[str,List] ):
        self.spec = spec
        self.keys = self.spec.keys()

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        return num
