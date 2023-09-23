from collections import OrderedDict

MAPPING = {
    255: (255, 255, 255),
    6: (0, 0, 63),
    9: (0, 191, 127),
    1: (0, 63, 0),
    7: (0, 63, 127),
    8: (0, 63, 191),
    3: (0, 63, 255),
    2: (0, 127, 63),
    5: (0, 127, 127),
    4: (0, 0, 127),
    14: (0, 0, 191),
    13: (0, 0, 255),
    11: (0, 63, 63),
    10: (0, 127, 191),
    0: (0, 127, 255),
    12: (0, 100, 155),
    15: (64, 191, 127),
    16: (64, 0, 191),
    17: (128, 63, 63),
    18: (128, 0, 63),
    19: (191, 63, 0),
    20: (255, 127, 0),
    21: (63, 0, 0),
    22: (127, 63, 0),
    23: (63, 255, 0),
    24: (0, 127, 0),
    25: (127, 127, 0),
    26: (63, 0, 63),
    27: (63, 127, 0),
    28: (63, 191, 0),
    29: (191, 127, 0),
    30: (127, 191, 0),
    31: (63, 63, 0),
    32: (100, 155, 0),
    33: (0, 255, 0),
    34: (0, 191, 0),
    35: (191, 127, 64),
    36: (0, 191, 64)
    }

# borrow from https://github.com/jbwang1997/BboxToolkit

DOTA2_0 = ('large-vehicle', 'swimming-pool', 'helicopter', 'bridge',
                'plane', 'ship', 'soccer-ball-field', 'basketball-court',
                'ground-track-field', 'small-vehicle', 'baseball-diamond',
                'tennis-court', 'roundabout', 'storage-tank', 'harbor',
                'container-crane', 'airport', 'helipad')

DIOR = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
        'chimney', 'expressway-service-area', 'expressway-toll-station',
        'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
        'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
        'windmill')

FAIR1M = ('A220','A321','A330','A350','ARJ21','Baseball-Field','Basketball-Court',
'Boeing737','Boeing747','Boeing777','Boeing787','Bridge','Bus','C919','Cargo-Truck',
'Dry-Cargo-Ship','Dump-Truck','Engineering-Ship','Excavator','Fishing-Boat',
'Football-Field','Intersection','Liquid-Cargo-Ship','Motorboat','other-airplane',
'other-ship','other-vehicle','Passenger-Ship','Roundabout','Small-Car','Tennis-Court',
'Tractor','Trailer','Truck-Tractor','Tugboat','Van','Warship')
