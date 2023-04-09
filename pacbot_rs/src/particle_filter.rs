use crate::grid::is_walkable;
use crate::variables::PACBOT_STARTING_POS;

use vecmath::{Vector2};

const PARTICLE_FILTER_POINTS: usize = 1000;

#[derive(Clone, Copy)]
struct PfPosition {
    x: f64,
    y: f64,
}

impl PfPosition {
    fn dist(&self, other: PfPosition) -> f64 {
        ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt()
    }
}

#[derive(Clone, Copy)]
struct PfPose {
    pos: PfPosition,
    angle: f64, // radians
}

pub struct ParticleFilter {
    pacbot_pos: PfPosition,
    points: [PfPose; PARTICLE_FILTER_POINTS],
    empty_grid_cells: Vec<PfPosition>
}

impl ParticleFilter {
    fn update_cell_sort(&mut self) {
        self.empty_grid_cells.sort_by(|a, b| a.dist(self.pacbot_pos)
            .total_cmp(&b.dist(self.pacbot_pos)));
    }

    fn new() -> Self {
        let empty_pose = PfPose {
            pos: PfPosition {x: 0.0, y: 0.0},
            angle: 0.0,
        };

        let empty_grid_cells = (0..28)
            .flat_map(|x| (0..31).map(move |y| (x, y)))
            .filter(|&pair| is_walkable(pair))
            .map(|pair| PfPosition {x: pair.0 as f64,  y: pair.1 as f64})
            .collect();

        let points = [empty_pose; PARTICLE_FILTER_POINTS];

        let mut pf = Self {
            pacbot_pos: PfPosition { x: PACBOT_STARTING_POS.0 as f64, y: PACBOT_STARTING_POS.1 as f64},
            points,
            empty_grid_cells
        };

        pf.update_cell_sort();

        pf
    }
}