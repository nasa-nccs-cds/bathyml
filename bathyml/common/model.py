from typing import List, Optional, Tuple, Dict, Any

class Model:

    def __init__(self, type, initParms: Dict[str,Any], parmRanges: Dict[str,List] ):
        self._type = type
        self._initParms = initParms
        self.parmRanges = parmRanges
        self._currentParms = { **initParms }

    def getInstance(self):
        if self._type == "svr": pass

