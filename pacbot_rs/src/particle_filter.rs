use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::grid::{GRID, is_walkable};
use crate::variables::{INNER_CELL_WIDTH, ROBOT_WIDTH};

const PARTICLE_FILTER_POINTS: usize = 1000;
const EPSILON: f64 = 0.0000001;

const SENSOR_ANGLES: [f64; 5] = [0.0,
    (1.0 / 8.0) * std::f64::consts::PI,
    (1.0 / 4.0) * std::f64::consts::PI,
    (-1.0 / 8.0) * std::f64::consts::PI,
    (-1.0 / 4.0) * std::f64::consts::PI];

const SENSOR_DISTANCE_FROM_CENTER: f64 = 0.75 / 2.0;

struct HorizontalSegment {
    x_min: f64,
    x_max: f64,
    y: f64,
}

impl HorizontalSegment {
    fn raycast(&self, x0: f64, y0: f64, vx: f64, vy: f64) -> Option<f64> {
        if vy.abs() < EPSILON {
            return None;
        }

        let distance = (self.y - y0) / vy;
        return if distance < 0.0 || (!(self.x_min <= x0 + vx * distance && x0 + vx * distance <= self.x_max)) {
            None
        } else {
            Some(distance)
        };
    }
}

struct VerticalSegment {
    x: f64,
    y_min: f64,
    y_max: f64,
}

impl VerticalSegment {
    fn raycast(&self, x0: f64, y0: f64, vx: f64, vy: f64) -> Option<f64> {
        if vx.abs() < EPSILON {
            return None;
        }

        let distance = (self.x - x0) / vx;
        return if distance < 0.0 || (!(self.y_min <= y0 + vy * distance && y0 + vy * distance <= self.y_max)) {
            None
        } else {
            Some(distance)
        };
    }
}

#[derive(Clone, Copy)]
struct PfPosition {
    x: f64,
    y: f64,
}

impl PfPosition {
    fn dist(&self, other: PfPosition) -> f64 {
        ((self.x - other.x).powf(2.0) + (self.y - other.y).powf(2.0)).sqrt()
    }

    fn update_by(&mut self, magnitude: f64, direction: f64) {
        self.x += magnitude * direction.cos();
        self.y += magnitude * direction.sin();
    }
}

#[derive(Clone, Copy)]
struct PfPose {
    pos: PfPosition,
    angle: f64, // radians
}

pub struct ParticleFilter {
    pacbot_pose: PfPose,
    points: [PfPose; PARTICLE_FILTER_POINTS],
    empty_grid_cells: Vec<PfPosition>,
    map_segments: (Vec<HorizontalSegment>, Vec<VerticalSegment>),
}

impl ParticleFilter {
    fn update_cell_sort(&mut self) {
        self.empty_grid_cells.sort_by(|a, b| a.dist(self.pacbot_pose.pos)
            .total_cmp(&b.dist(self.pacbot_pose.pos)));
    }

    fn random_point(&self) -> PfPose {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 10.0).unwrap();
        let random_value = rng.sample::<f64, _>(normal).abs();
        let index = random_value.round() as usize;

        let pos = self.empty_grid_cells[index].clone();

        let extra_space_per_side = (ROBOT_WIDTH - INNER_CELL_WIDTH) / 2.0;

        let x_range = Uniform::new(-extra_space_per_side, extra_space_per_side);
        let y_range = Uniform::new(-extra_space_per_side, extra_space_per_side);

        let random_x = x_range.sample(&mut rng);
        let random_y = y_range.sample(&mut rng);

        let pos = PfPosition {
            x: pos.x + random_x,
            y: pos.y + random_y,
        };

        let angle_range = Uniform::new(0.0, 2.0 * std::f64::consts::PI);
        let angle = angle_range.sample(&mut rng);

        PfPose { pos, angle }
    }

    fn get_map_segments() -> (Vec<HorizontalSegment>, Vec<VerticalSegment>) {
        let mut horizontal_segments = Vec::new();
        let mut vertical_segments = Vec::new();

        // return the segments that represent walls in the map
        let grid_width = 28;
        let grid_height = 31;

        for y in 0..grid_height - 1 {
            let mut seg_start_x: Option<usize> = None;
            for x in 0..grid_width {
                let is_wall_here = GRID[x][y] != GRID[x][y + 1];
                if is_wall_here && seg_start_x == None {
                    seg_start_x = Some(x - 1);
                }
                if !is_wall_here && seg_start_x != None {
                    horizontal_segments.push(HorizontalSegment {
                        x_min: seg_start_x.unwrap() as f64,
                        x_max: (x - 1) as f64,
                        y: y as f64,
                    });
                    seg_start_x = None;
                }
            }
        }

        for x in 0..grid_width - 1 {
            let mut seg_start_y: Option<usize> = None;
            for y in 0..grid_height {
                let is_wall_here = GRID[x][y] != GRID[x + 1][y];
                if is_wall_here && seg_start_y == None {
                    seg_start_y = Some(y - 1);
                }
                if !is_wall_here && seg_start_y != None {
                    vertical_segments.push(VerticalSegment {
                        x: x as f64,
                        y_min: seg_start_y.unwrap() as f64,
                        y_max: (y - 1) as f64,
                    });
                    seg_start_y = None;
                }
            }
        }

        (horizontal_segments, vertical_segments)
    }

    fn raycast(&self, start_pos: PfPosition, angle: f64) -> f64 {
        let (horizontal_segments, vertical_segments) = &self.map_segments;

        let mut min_distance = f64::INFINITY;

        let vx = angle.cos();
        let vy = angle.sin();

        for segment in horizontal_segments {
            if let Some(distance) = segment.raycast(start_pos.x, start_pos.y, vx, vy) {
                min_distance = min_distance.min(distance);
            }
        }

        for segment in vertical_segments {
            if let Some(distance) = segment.raycast(start_pos.x, start_pos.y, vx, vy) {
                min_distance = min_distance.min(distance);
            }
        }

        min_distance
    }

    pub fn new(pacbot_pos: (usize, usize, f64)) -> Self {
        let empty_pose = PfPose {
            pos: PfPosition { x: 0.0, y: 0.0 },
            angle: 0.0,
        };

        let empty_grid_cells = (0..28)
            .flat_map(|x| (0..31).map(move |y| (x, y)))
            .filter(|&pair| is_walkable(pair))
            .map(|pair| PfPosition { x: pair.0 as f64, y: pair.1 as f64 })
            .collect();

        let points = [empty_pose; PARTICLE_FILTER_POINTS];

        let mut pf = Self {
            pacbot_pose: PfPose {
                pos: PfPosition {
                    x: pacbot_pos.0 as f64,
                    y: pacbot_pos.1 as f64,
                },
                angle: pacbot_pos.2,
            },
            points,
            empty_grid_cells,
            map_segments: Self::get_map_segments(),
        };

        pf.update_cell_sort();

        // set points to random positions
        for i in 0..PARTICLE_FILTER_POINTS {
            pf.points[i] = pf.random_point();
        }

        pf
    }

    fn get_point_accuracy(&self, point: &PfPose, sensors: [f64; 5]) -> f64 {
        let mut accuracy = 0.0;

        for i in 0..5 {
            let angle = point.angle + SENSOR_ANGLES[i];
            let distance = self.raycast(point.pos, angle);
            let sensor_distance = sensors[i];

            let diff = (distance - sensor_distance).abs();

            accuracy += diff;
        }

        accuracy
    }

    pub fn update(&mut self, magnitude: f64, direction: f64, sensors: [f64; 5]) {
        for point in self.points.iter_mut() {
            point.pos.update_by(magnitude, direction);
            point.angle += direction;
        }

        // sort points by accuracy based on sensors
        let mut point_accuracies: Vec<(&PfPose, f64)> = self.points
            .iter()
            .map(|point| (point, self.get_point_accuracy(point, sensors)))
            .collect();

        point_accuracies.sort_by(|(_, a_accuracy), (_, b_accuracy)| {
            a_accuracy.partial_cmp(b_accuracy).unwrap()
        });

        let sorted_points: Vec<PfPose> = point_accuracies.into_iter().map(|(point, _)| *point).collect();
        self.points.copy_from_slice(&sorted_points);

        // find the best guess position
        self.pacbot_pose = self.points[0];
        self.update_cell_sort();

        // replace the worst half of the points with random points around the best points
        for i in 0..PARTICLE_FILTER_POINTS / 2 {
            // choose a random index from 0 to PARTICLE_FILTER_POINTS, weighing lower values more
            let index_range = Uniform::new(0.0, ((PARTICLE_FILTER_POINTS / 2) as f64).sqrt());
            let mut index = index_range.sample(&mut rand::thread_rng()).powf(2.0) as usize;
            if index >= PARTICLE_FILTER_POINTS {
                index = PARTICLE_FILTER_POINTS - 1;
            }

            let old_point = self.points[index];
            // choose a random small angle and distance from the old point
            let angle_range = Uniform::new(-0.1, 0.1);
            let distance_range = Uniform::new(-0.1, 0.1);
            let angle: f64 = angle_range.sample(&mut rand::thread_rng());
            let distance: f64 = distance_range.sample(&mut rand::thread_rng());

            self.points[i + PARTICLE_FILTER_POINTS / 2] = PfPose {
                pos: PfPosition {
                    x: old_point.pos.x + angle.cos() * distance,
                    y: old_point.pos.y + angle.sin() * distance,
                },
                angle: old_point.angle + angle,
            };
        }

        // replace some of the new points with completely random points
        for i in 0..PARTICLE_FILTER_POINTS / 10 {
            self.points[i] = self.random_point();
        }
    }
}