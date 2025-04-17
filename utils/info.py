import math
from typing import Tuple

import torch

OBJECT_NAME2ID = {
    # seen category
    "Box": 0,
    "Remote": 1,
    "Microwave": 2,
    "Camera": 3,
    "Dishwasher": 4,
    "WashingMachine": 5,
    "CoffeeMachine": 6,
    "Toaster": 7,
    "StorageFurniture": 8,
    "AKBBucket": 9, # akb48
    "AKBBox": 10, # akb48
    "AKBDrawer": 11, # akb48
    "AKBTrashCan": 12, # akb48
    "Bucket": 13, # new
    "Keyboard": 14, # new
    "Printer": 15, # new
    "Toilet": 16, # new
    # unseen category
    "KitchenPot": 17,
    "Safe": 18,
    "Oven": 19,
    "Phone": 20,
    "Refrigerator": 21,
    "Table": 22,
    "TrashCan": 23,
    "Door": 24,
    "Laptop": 25,
    "Suitcase": 26, # new
}

TARGET_PARTS = [
    'others',
    'line_fixed_handle',
    'round_fixed_handle',
    'slider_button',
    'hinge_door',
    'slider_drawer',
    'slider_lid',
    'hinge_lid',
    'hinge_knob',
    'revolute_handle'
]

PART_NAME2ID = {
    'others':             0,
    'line_fixed_handle':  1,
    'round_fixed_handle': 2,
    'slider_button':      3,
    'hinge_door':         4,
    'slider_drawer':      5,
    'slider_lid':         6,
    'hinge_lid':          7,
    'hinge_knob':         8,
    'revolute_handle':    9,
}

PART_ID2NAME = {
    0: 'others'             ,
    1: 'line_fixed_handle'  ,
    2: 'round_fixed_handle' ,
    3: 'slider_button'      ,
    4: 'hinge_door'         ,
    5: 'slider_drawer'      ,
    6: 'slider_lid'         ,
    7: 'hinge_lid'          ,
    8: 'hinge_knob'         ,
    9: 'revolute_handle'    ,
}


TARGET_PARTS = [
    'others',
    'line_fixed_handle',
    'round_fixed_handle',
    'slider_button',
    'hinge_door',
    'slider_drawer',
    'slider_lid',
    'hinge_lid',
    'hinge_knob',
    'revolute_handle',
]

TARGET_IDX = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
]
PI = math.pi
SYMMETRY_MATRIX = [
    # type 0
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
    ],

    # type 1
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [-1.0,     0,     0],
            [ 0,    -1.0,     0],
            [ 0,       0,   1.0],
        ],
    ],

    # type 2
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [-1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,  -1.0],
        ],
    ],

    # type 3
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [ math.cos(PI/6), math.sin(PI/6),          0],
            [-math.sin(PI/6), math.cos(PI/6),          0],
            [              0,              0,        1.0]
        ],
        [
            [ math.cos(PI*2/6), math.sin(PI*2/6),          0],
            [-math.sin(PI*2/6), math.cos(PI*2/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*3/6), math.sin(PI*3/6),          0],
            [-math.sin(PI*3/6), math.cos(PI*3/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*4/6), math.sin(PI*4/6),          0],
            [-math.sin(PI*4/6), math.cos(PI*4/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*5/6), math.sin(PI*5/6),          0],
            [-math.sin(PI*5/6), math.cos(PI*5/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*6/6), math.sin(PI*6/6),         0],
            [-math.sin(PI*6/6), math.cos(PI*6/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*7/6), math.sin(PI*7/6),         0],
            [-math.sin(PI*7/6), math.cos(PI*7/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*8/6), math.sin(PI*8/6),         0],
            [-math.sin(PI*8/6), math.cos(PI*8/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*9/6), math.sin(PI*9/6),         0],
            [-math.sin(PI*9/6), math.cos(PI*9/6),         0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*10/6), math.sin(PI*10/6),         0],
            [-math.sin(PI*10/6), math.cos(PI*10/6),         0],
            [                 0,                 0,        1.0]
        ],
        [
            [ math.cos(PI*11/6), math.sin(PI*11/6),         0],
            [-math.sin(PI*11/6), math.cos(PI*11/6),         0],
            [                 0,                 0,        1.0]
        ],
    ],

    # type 4
    [
        [
            [ 1.0,     0,     0],
            [ 0,     1.0,     0],
            [ 0,       0,   1.0],
        ],
        [
            [ math.cos(PI/6), math.sin(PI/6),          0],
            [-math.sin(PI/6), math.cos(PI/6),          0],
            [              0,              0,        1.0]
        ],
        [
            [ math.cos(PI*2/6), math.sin(PI*2/6),          0],
            [-math.sin(PI*2/6), math.cos(PI*2/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*3/6), math.sin(PI*3/6),          0],
            [-math.sin(PI*3/6), math.cos(PI*3/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*4/6), math.sin(PI*4/6),          0],
            [-math.sin(PI*4/6), math.cos(PI*4/6),          0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*5/6), math.sin(PI*5/6),        0],
            [-math.sin(PI*5/6), math.cos(PI*5/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*6/6), math.sin(PI*6/6),        0],
            [-math.sin(PI*6/6), math.cos(PI*6/6),        0],
            [                0,               0,        1.0]
        ],
        [
            [ math.cos(PI*7/6), math.sin(PI*7/6),        0],
            [-math.sin(PI*7/6), math.cos(PI*7/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*8/6), math.sin(PI*8/6),        0],
            [-math.sin(PI*8/6), math.cos(PI*8/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*9/6), math.sin(PI*9/6),        0],
            [-math.sin(PI*9/6), math.cos(PI*9/6),        0],
            [                0,                0,        1.0]
        ],
        [
            [ math.cos(PI*10/6), math.sin(PI*10/6),        0],
            [-math.sin(PI*10/6), math.cos(PI*10/6),        0],
            [                 0,                 0,        1.0]
        ],
        [
            [ math.cos(PI*11/6), math.sin(PI*11/6),        0],
            [-math.sin(PI*11/6), math.cos(PI*11/6),        0],
            [                 0,                 0,        1.0]
        ],
        ######################  inverse  ######################
        [
            [ math.sin(PI/6), math.cos(PI/6),        0],
            [ math.cos(PI/6), -math.sin(PI/6),        0],
            [              0,              0,       -1.0]
        ],
        [
            [ math.sin(PI*2/6), math.cos(PI*2/6),        0],
            [ math.cos(PI*2/6), -math.sin(PI*2/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*3/6), math.cos(PI*3/6),        0],
            [ math.cos(PI*3/6), -math.sin(PI*3/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*4/6), math.cos(PI*4/6),        0],
            [ math.cos(PI*4/6), -math.sin(PI*4/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*5/6), math.cos(PI*5/6),        0],
            [ math.cos(PI*5/6), -math.sin(PI*5/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*6/6), math.cos(PI*6/6),        0],
            [ math.cos(PI*6/6), -math.sin(PI*6/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*7/6), math.cos(PI*7/6),        0],
            [ math.cos(PI*7/6), -math.sin(PI*7/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*8/6), math.cos(PI*8/6),        0],
            [ math.cos(PI*8/6), -math.sin(PI*8/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*9/6), math.cos(PI*9/6),        0],
            [ math.cos(PI*9/6), -math.sin(PI*9/6),        0],
            [                0,                0,       -1.0]
        ],
        [
            [ math.sin(PI*10/6), math.cos(PI*10/6),        0],
            [ math.cos(PI*10/6), -math.sin(PI*10/6),        0],
            [                 0,                 0,       -1.0]
        ],
        [
            [ math.sin(PI*11/6), math.cos(PI*11/6),        0],
            [ math.cos(PI*11/6), -math.sin(PI*11/6),        0],
            [                 0,                 0,       -1.0]
        ],
        [
            [ math.sin(PI*12/6), math.cos(PI*12/6),        0],
            [ math.cos(PI*12/6), -math.sin(PI*12/6),        0],
            [                 0,                 0,       -1.0]
        ],
    ],
]


def get_symmetry_matrix() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # type 0 / 1 / 2
    sm_1 = torch.as_tensor(SYMMETRY_MATRIX[:3], dtype=torch.float32)
    # type 3
    sm_2 = torch.as_tensor(SYMMETRY_MATRIX[3:4], dtype=torch.float32)
    # type 4
    sm_3 = torch.as_tensor(SYMMETRY_MATRIX[4:5], dtype=torch.float32)

    return sm_1, sm_2, sm_3
