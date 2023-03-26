use num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Direction {
    Right = 0,
    Left = 1,
    Up = 2,
    Down = 3,
}

/// Enum for grid cell values.
#[derive(Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
#[allow(non_camel_case_types)]
pub enum GridValue {
    /// Wall
    I = 1,
    /// Normal pellet
    o = 2,
    /// Empty space
    e = 3,
    /// Power pellet
    O = 4,
    /// Ghost chambers
    n = 5,
    /// Cherry position
    c = 6,
}

/*
# State enums
scatter = 1
chase = 2
frightened = 3

# color enums
red = 1
orange = 2
pink = 3
blue = 4

# input signal enums
sig_normal = 0
sig_quit = 1
sig_restart = 2

# game Params
starting_lives = 3
frightened_length = 40
pellet_score = 10
power_pellet_score = 50
cherry_score = 100
ghost_score = 200
state_swap_times = [35, 135, 170, 270, 295, 395, 420]
*/
pub const PACBOT_STARTING_POS: (usize, usize) = (14, 7);
pub const PACBOT_STARTING_DIR: Direction = Direction::Left;
/*
cherry_pos = (13, 13)
game_frequency = 2.0
ticks_per_update = 12
*/
