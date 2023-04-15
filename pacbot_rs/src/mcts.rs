use num_enum::TryFromPrimitive;
use ordered_float::NotNan;
use pyo3::prelude::*;

use crate::game_state::env::{Action, PacmanGym};

/// The type for returns (cumulative rewards).
type Return = f32;

#[derive(Default)]
struct SearchTreeEdge {
    visit_count: u32,
    total_return: Return,
    child: Option<Box<SearchTreeNode>>,
}

impl SearchTreeEdge {
    /// Returns the estimated expected return (cumulative reward) for this action.
    #[must_use]
    pub fn expected_return(&self) -> NotNan<f32> {
        if self.visit_count == 0 {
            NotNan::new(0.0).unwrap() // TODO: handle this case in a more principled way?
        } else {
            let expected_return = self.total_return / (self.visit_count as f32);
            NotNan::new(expected_return).expect("expected score is NaN")
        }
    }

    /// A variant of the PUCT score, similar to that used in AlphaZero.
    #[must_use]
    pub fn puct_score(&self, parent_visit_count: u32) -> NotNan<f32> {
        let exploration_rate = 100.0; // TODO: make this a tunable parameter
        let exploration_score =
            exploration_rate * (parent_visit_count as f32).sqrt() / ((1 + self.visit_count) as f32);
        self.expected_return() + exploration_score
    }
}

#[derive(Default)]
struct SearchTreeNode {
    visit_count: u32,
    children: [SearchTreeEdge; 5],
}

impl SearchTreeNode {
    /// Samples a move that a player might make from a state, updating the search tree.
    /// Mutates the provided environment instance as the tree walk is performed.
    /// Returns the return (cumulative reward; based on the search steps taken and the
    /// leaf evaluation).
    fn sample_move(
        &mut self,
        env: &mut PacmanGym,
        leaf_evaluator: impl FnOnce(&mut PacmanGym) -> Return,
    ) -> Return {
        // choose a (valid) action based on the current stats
        let action_mask = env.action_mask();
        let (action_index, edge) = self
            .children
            .iter_mut()
            .enumerate()
            .filter(|&(action_index, _)| action_mask[action_index])
            .max_by_key(|(_, edge)| edge.puct_score(self.visit_count))
            .unwrap();
        let action = Action::try_from_primitive(action_index.try_into().unwrap()).unwrap();

        // update the environment and recurse / evaluate the leaf
        let (reward, done) = env.step(action);
        let subsequent_return = if done {
            // this child is a terminal node; the return is therefore zero
            0.0
        } else if let Some(child) = &mut edge.child {
            // this child has already been expanded; recurse
            child.sample_move(env, leaf_evaluator)
        } else {
            // this child is a leaf node; expand it and evaluate it
            edge.child = Some(Default::default());
            leaf_evaluator(env)
        };
        let this_return = reward as Return + subsequent_return;

        // update the stats for this action
        self.visit_count += 1;
        edge.visit_count += 1;
        edge.total_return += this_return;

        this_return
    }

    /// Returns the valid action with the highest expected return.
    fn best_action(&self, env: &PacmanGym) -> Action {
        let action_mask = env.action_mask();
        let (action_index, _) = self
            .children
            .iter()
            .enumerate()
            .filter(|&(action_index, _)| action_mask[action_index])
            .max_by_key(|(_, edge)| edge.expected_return())
            .unwrap();
        Action::try_from_primitive(action_index.try_into().unwrap()).unwrap()
    }

    fn max_depth(&self) -> usize {
        self.children
            .iter()
            .filter_map(|edge| edge.child.as_ref())
            .map(|child| 1 + child.max_depth())
            .max()
            .unwrap_or(0)
    }
}

#[pyclass]
pub struct MCTSContext {
    root: SearchTreeNode,
}

#[pymethods]
impl MCTSContext {
    #[new]
    pub fn new() -> Self {
        Self {
            root: SearchTreeNode::default(),
        }
    }

    /// Updates the search tree root to the node resulting from the given action.
    pub fn update_root(&mut self, action: Action) {
        self.root = self.root.children[u8::from(action) as usize]
            .child
            .take()
            .map(|child_box| *child_box)
            .unwrap_or_default();
    }

    /// Returns the action at the root with the highest expected return.
    pub fn best_action(&self, env: &PacmanGym) -> Action {
        self.root.best_action(env)
    }

    /// Performs the specified number of MCTS iterations, then returns the best action.
    pub fn ponder_and_choose(&mut self, env: &PacmanGym, num_iterations: usize) -> Action {
        for _ in 0..num_iterations {
            self.sample_move(env.clone(), |_env| 0.0);
        }
        self.best_action(env)
    }
}

impl MCTSContext {
    /// Samples a move that a player might make from a state, updating the search tree.
    /// Returns the return (cumulative reward; based on the search steps taken and the
    /// leaf evaluation).
    pub fn sample_move(
        &mut self,
        mut env: PacmanGym,
        leaf_evaluator: impl FnOnce(&mut PacmanGym) -> Return,
    ) -> Return {
        if env.is_done() {
            0.0
        } else {
            self.root.sample_move(&mut env, leaf_evaluator)
        }
    }
}

impl Default for MCTSContext {
    fn default() -> Self {
        Self::new()
    }
}