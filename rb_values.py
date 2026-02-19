from enum import Enum
from typing import Mapping, Optional


class SCSkHz(Enum):
    KHZ_15 = 15
    KHZ_30 = 30
    KHZ_60 = 60


class BWMHz(Enum):
    MHZ_5 = 5
    MHZ_10 = 10
    MHZ_15 = 15
    MHZ_20 = 20
    MHZ_25 = 25
    MHZ_30 = 30
    MHZ_40 = 40
    MHZ_50 = 50
    MHZ_60 = 60
    MHZ_80 = 80
    MHZ_90 = 90
    MHZ_100 = 100


# ---- Table values (NRB) ----
NRB_TABLE: Mapping[SCSkHz, Mapping[BWMHz, Optional[int]]] = {
    SCSkHz.KHZ_15: {
        BWMHz.MHZ_5: 25,
        BWMHz.MHZ_10: 52,
        BWMHz.MHZ_15: 79,
        BWMHz.MHZ_20: 106,
        BWMHz.MHZ_25: 133,
        BWMHz.MHZ_30: 160,
        BWMHz.MHZ_40: 216,
        BWMHz.MHZ_50: 270,
        BWMHz.MHZ_60: None,
        BWMHz.MHZ_80: None,
        BWMHz.MHZ_90: None,
        BWMHz.MHZ_100: None,
    },
    SCSkHz.KHZ_30: {
        BWMHz.MHZ_5: 11,
        BWMHz.MHZ_10: 24,
        BWMHz.MHZ_15: 38,
        BWMHz.MHZ_20: 51,
        BWMHz.MHZ_25: 65,
        BWMHz.MHZ_30: 78,
        BWMHz.MHZ_40: 106,
        BWMHz.MHZ_50: 133,
        BWMHz.MHZ_60: 162,
        BWMHz.MHZ_80: 217,
        BWMHz.MHZ_90: 245,
        BWMHz.MHZ_100: 273,
    },
    SCSkHz.KHZ_60: {
        BWMHz.MHZ_5: None,
        BWMHz.MHZ_10: 11,
        BWMHz.MHZ_15: 18,
        BWMHz.MHZ_20: 24,
        BWMHz.MHZ_25: 31,
        BWMHz.MHZ_30: 38,
        BWMHz.MHZ_40: 51,
        BWMHz.MHZ_50: 65,
        BWMHz.MHZ_60: 79,
        BWMHz.MHZ_80: 107,
        BWMHz.MHZ_90: 121,
        BWMHz.MHZ_100: 135,
    },
}
