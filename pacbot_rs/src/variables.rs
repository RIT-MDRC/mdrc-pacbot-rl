use num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum Direction {
    Right = 0,
    Left = 1,
    Up = 2,
    Down = 3,
}

/// Enum for grid cell values.
#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
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
# input signal enums
sig_normal = 0
sig_quit = 1
sig_restart = 2
*/

// game params
// pub const starting_lives = 3;
pub const FRIGHTENED_LENGTH: u32 = 40;
// pub const pellet_score = 10;
// pub const power_pellet_score = 50;
// pub const cherry_score = 100;
// pub const ghost_score = 200;
// pub const state_swap_times = [35, 135, 170, 270, 295, 395, 420];
pub const PACBOT_STARTING_POS: (usize, usize) = (14, 7);
pub const PACBOT_STARTING_DIR: Direction = Direction::Left;
/*
cherry_pos = (13, 13)
game_frequency = 2.0
ticks_per_update = 12
*/
