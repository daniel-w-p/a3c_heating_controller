from enum import Enum


class AppMode(Enum):
    LEARN = 1
    RUN = 2
    COMPARE = 3


gui = {
    'SCREEN_WIDTH': 1000,
    'SCREEN_HEIGHT': 750,
    'WINDOW_TITLE': 'Heating Controller',
    'FRAME_RATE': 1,

}

ai = {
    'RUN_MODE': AppMode.RUN,
}
