use num_enum::{IntoPrimitive, TryFromPrimitive};

use crate::{ghost_agent::GhostAgent, pacbot::PacBot, variables::GridValue};

#[derive(Clone, Copy, Debug, Eq, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub enum GameStateState {
    Scatter = 1,
    Chase = 2,
    Frightened = 3,
}

pub struct GameState {
    pub pacbot: PacBot,

    pub red: GhostAgent,
    pub pink: GhostAgent,
    pub orange: GhostAgent,
    pub blue: GhostAgent,

    pub just_swapped_state: bool,
    pub state: GameStateState,
    pub start_counter: usize,

    pub grid: [[GridValue; 31]; 28],
}
