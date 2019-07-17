import logging, os, time, socket
LOG_DIR=os.path.expanduser("~/.bathyml/logs")

class BathmlLogger:
    logger = None

    @classmethod
    def getLogger(cls):
        if cls.logger is None:
            if not os.path.exists(LOG_DIR):  os.makedirs(LOG_DIR)
            timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
            cls.logger = logging.getLogger( "bathyml" )
            cls.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler("{}/bathyml-{}-{}.log".format(LOG_DIR, socket.gethostname(), timestamp))
            fh.setLevel(logging.DEBUG)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('bathyml-%(asctime)s-%(levelname)s: %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            cls.logger.addHandler(fh)
            cls.logger.addHandler(ch)
        return cls.logger
