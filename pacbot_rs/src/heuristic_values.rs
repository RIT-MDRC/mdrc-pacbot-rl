use std::collections::{HashSet, VecDeque};

use ordered_float::NotNan;
use pyo3::prelude::*;

use crate::{
    game_state::{
        env::{Action, PacmanGym},
        GameState,
    },
    grid::{self, coords_to_node, DISTANCE_MATRIX, SUPER_PELLET_LOCS},
    mcts::MCTSContext,
    variables::GridValue,
};

/// Performs a breadth-first search through the grid. Returns the distance from
/// the given start position to the closest position for which is_goal(pos) is
/// true, or returns None if no such position is reachable.
fn breadth_first_search(
    start: (usize, usize),
    mut is_goal: impl FnMut((usize, usize)) -> bool,
) -> Option<usize> {
    if is_goal(start) {
        return Some(0);
    }

    let mut queue = VecDeque::from([(start, 0)]);
    let mut visited = HashSet::from([start]);
    while let Some((cur_pos, cur_dist)) = queue.pop_front() {
        let neighbor_dist = cur_dist + 1;
        let (cx, cy) = cur_pos;
        for neighbor in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)] {
            if grid::is_walkable(neighbor) && visited.insert(neighbor) {
                if is_goal(neighbor) {
                    return Some(neighbor_dist);
                }
                queue.push_back((neighbor, neighbor_dist));
            }
        }
    }
    None
}

const FEAR: u8 = 10;
const PELLET_WEIGHT: f32 = 0.65;
const SUPER_PELLET_WEIGHT: f32 = 10.0;
const GHOST_WEIGHT: f32 = 0.35;
const FRIGHTENED_GHOST_WEIGHT: f32 = 0.3 * GHOST_WEIGHT;

/// Computes the heuristic value from the hand-coded algorithm for the given
/// position in the game grid. Returns None if the given position is not walkable.
#[pyfunction]
pub fn get_heuristic_value(game_state: &GameState, pos: (usize, usize)) -> Option<f32> {
    let pos_node = coords_to_node(pos)?;

    let pellet_dist =
        breadth_first_search(pos, |(x, y)| game_state.grid[x][y] == GridValue::o).unwrap_or(0);
    let pellet_heuristic = pellet_dist as f32 * PELLET_WEIGHT;

    let super_pellet_dist = SUPER_PELLET_LOCS
        .iter()
        .filter(|&&(x, y)| game_state.grid[x][y] == GridValue::O)
        .map(|&pos| DISTANCE_MATRIX[pos_node][coords_to_node(pos).unwrap()])
        .min()
        .unwrap_or(0);
    let super_pellet_heuristic = super_pellet_dist as f32 * SUPER_PELLET_WEIGHT;

    // get the distance and frightened status of all (reachable) ghosts
    let ghost_dists = [
        game_state.red.borrow(),
        game_state.pink.borrow(),
        game_state.orange.borrow(),
        game_state.blue.borrow(),
    ]
    .into_iter()
    .filter_map(|ghost| {
        coords_to_node(ghost.current_pos)
            .map(|node| (DISTANCE_MATRIX[pos_node][node], ghost.is_frightened()))
    })
    .filter(|(dist, _)| *dist < FEAR);

    let ghost_heuristic: f32 = ghost_dists
        .into_iter()
        .map(|(dist, is_frightened)| {
            let weight = if is_frightened {
                -FRIGHTENED_GHOST_WEIGHT
            } else {
                GHOST_WEIGHT
            };
            let fear_minus_dist = FEAR - dist;
            (fear_minus_dist * fear_minus_dist) as f32 * weight
        })
        .sum();

    Some(pellet_heuristic + super_pellet_heuristic + ghost_heuristic)
}

/// Computes the heuristic values for each of the 5 actions for the given GameState.
/// Returns the values as well as the best action.
#[pyfunction]
pub fn get_action_heuristic_values(game_state: &GameState) -> ([Option<f32>; 5], u8) {
    let (px, py) = game_state.pacbot.pos;

    let values = [
        (px, py),
        (px, py + 1),
        (px, py - 1),
        (px - 1, py),
        (px + 1, py),
    ]
    .map(|pos| get_heuristic_value(game_state, pos));

    let best_action = values
        .iter()
        .enumerate()
        .filter_map(|(i, value)| value.map(|v| (i, v)))
        .min_by_key(|&(_, value)| NotNan::new(value).unwrap())
        .expect("at least one action should be valid")
        .0;

    (values, best_action.try_into().unwrap())
}

/// Computes a path from pacbot's current location, following the local heuristic
/// gradient towards a local minimum.
#[pyfunction]
pub fn get_heuristic_path(
    game_state: &GameState,
    max_path_len: Option<usize>,
) -> Vec<(usize, usize)> {
    let (mut x, mut y) = game_state.pacbot.pos;
    let mut last_action = None;

    let mut path = vec![(x, y)];

    loop {
        // check if we've reached the max_path_len
        if let Some(max_path_len) = max_path_len {
            if path.len() == max_path_len {
                break;
            }
        }

        // get the next point that has the best (lowest) heuristic value
        let next_points = [(x, y), (x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)];
        let values = next_points
            .iter()
            .map(|&pos| get_heuristic_value(game_state, pos));
        let min = values
            .enumerate()
            .zip(next_points)
            .filter_map(|((i, value), pos)| value.map(|v| (i, v, pos)))
            .min_by_key(|&(_, value, _)| NotNan::new(value).unwrap());
        let (next_action, _, best_next_point) = if let Some(min) = min {
            min
        } else {
            eprintln!("WARNING: get_heuristic_path: no valid actions");
            break;
        };

        // stop if we're at a local minimum; otherwise update the path and continue
        if best_next_point == (x, y) {
            break;
        } else {
            // add the current position to the path, replacing the last point if it's collinear
            if last_action == Some(next_action) {
                *path.last_mut().unwrap() = best_next_point;
            } else {
                path.push(best_next_point);
            }

            (x, y) = best_next_point;
            last_action = Some(next_action);
        }
    }

    path
}

/// Computes a path from pacbot's current location, following the moves that MCTS wants to make.
/// Params:
///     mcts_iterations: Number of MCTS iterations to perform per step. Defaults to 100.
#[pyfunction]
pub fn get_mcts_path(
    game_state: &GameState,
    max_path_len: Option<usize>,
    mcts_iterations: Option<usize>,
) -> Vec<(usize, usize)> {
    let mcts_iterations = mcts_iterations.unwrap_or(100);

    // initialize the environment to the current game_state
    let mut env = PacmanGym::new(false);
    env.game_state = game_state.clone();

    let mut path = vec![env.game_state.pacbot.pos];
    let mut last_action = None;

    loop {
        // check if we've reached the max_path_len
        if let Some(max_path_len) = max_path_len {
            if path.len() == max_path_len {
                break;
            }
        }

        // get the next action
        let mut mcts_context = MCTSContext::new();
        let action = mcts_context.ponder_and_choose(&env, mcts_iterations);

        // stop if the next action is to not move
        if action == Action::Stay {
            break;
        }

        // update the environment
        let (_, done) = env.step(action);

        // stop if the environment terminated; otherwise update the path and continue
        if done {
            break;
        } else {
            // add the current position to the path, replacing the last point if it's collinear
            let next_point = env.game_state.pacbot.pos;
            if last_action == Some(action) {
                *path.last_mut().unwrap() = next_point;
            } else {
                path.push(next_point);
            }

            last_action = Some(action);
        }
    }

    path
}
