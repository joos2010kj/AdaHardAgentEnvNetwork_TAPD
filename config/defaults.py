from fvcore.common.config import CfgNode

_C = CfgNode()

_C.TRAIN = CfgNode()
_C.TRAIN.BATCH_SIZE = 16
_C.TRAIN.LR = 0.001
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

_C.DATA = CfgNode()
_C.DATA.ENV_FEATURE_DIR = ""
_C.DATA.AGENT_FEATURE_DIR = ""
_C.DATA.VIDEO_ID_FILE = ""
_C.DATA.VIDEO_ANNOTATION_FILE = ""

_C.MODEL = CfgNode()
_C.MODEL.FEATURE_DIM = 2304

_C.BMN = CfgNode()
_C.BMN.NUM_SAMPLES = 32
_C.BMN.SOFT_NMS_ALPHA = 0.4
_C.BMN.SOFT_NMS_LOW_THRESHOLD = 0.5
_C.BMN.SOFT_NMS_HIGH_THRESHOLD = 0.9
