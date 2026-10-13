import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_1_3B import i2v_1_3B
from .wan_i2v_A14B import i2v_A14B

WAN_CONFIGS = {
    'i2v-A14B': i2v_A14B,
    'i2v-1.3B': i2v_1_3B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704),
    '1024*704': (1024, 704),
    '704*1024': (704, 1024),
    '960*1664': (960, 1664),
    '768*1152': (768, 1152),
    # 480*768 exists for cp8: its forced grid is 30h x 48w, where L = 1440 and gw = 48
    # BOTH divide by 8, so flat and width sharding agree. See exact_grids.py.
    '480*768': (480, 768),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
    '1024*704': 1024 * 704,
    '704*1024': 704 * 1024,
    '960*1664': 960 * 1664,
    '768*1152': 768 * 1152,
    '480*768': 480 * 768,
}

SUPPORTED_SIZES = {
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480', '960*1664',
                 '768*1152', '480*768'),
    'i2v-1.3B': ('720*1280', '1280*720', '480*832', '832*480'),
}


from .exact_grids import EXACT_GRID_SIZES  # noqa: F401  (re-export)
