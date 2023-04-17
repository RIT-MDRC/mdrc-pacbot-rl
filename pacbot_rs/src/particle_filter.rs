use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use crate::grid::{GRID, is_walkable, NODE_COORDS, NUM_NODES};
use crate::variables::{INNER_CELL_WIDTH, ROBOT_WIDTH};
use pyo3::prelude::*;

use ordered_float::NotNan;

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
    fn dist(&self, other: &PfPosition) -> f64 {
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

#[pyclass]
pub struct ParticleFilter {
    pacbot_pose: PfPose,
    points: [PfPose; PARTICLE_FILTER_POINTS],
    empty_grid_cells: [PfPosition; NUM_NODES],
    map_segments: (Vec<HorizontalSegment>, Vec<VerticalSegment>),
}

#[pymethods]
impl ParticleFilter {
    pub fn get_points(&self) -> Vec<((f64, f64), f64)> {
        self.points.iter().map(|p| ((p.pos.x, p.pos.y), p.angle)).collect()
    }

    pub fn get_empty_grid_cells(&self) -> Vec<(f64, f64)> {
        self.empty_grid_cells.iter().map(|p| (p.x, p.y)).collect()
    }

    pub fn get_map_segments_list(&self) -> Vec<(f64, f64, f64, f64)> {
        let (horiz, vert) = &self.map_segments;
        let mut segments = Vec::new();
        for h in horiz {
            segments.push((h.x_min, h.x_max, h.y, h.y));
        }
        for v in vert {
            segments.push((v.x, v.x, v.y_min, v.y_max));
        }
        segments
    }

    #[new]
    pub fn new(pacbot_x: usize, pacbot_y: usize, pacbot_angle: f64) -> Self {
        let empty_pose = PfPose {
            pos: PfPosition { x: 0.0, y: 0.0 },
            angle: 0.0,
        };

        let empty_grid_cells = NODE_COORDS.map(|(x, y)| PfPosition {
            x: x as f64,
            y: y as f64,
        });

        let points = [empty_pose; PARTICLE_FILTER_POINTS];

        let mut pf = Self {
            pacbot_pose: PfPose {
                pos: PfPosition {
                    x: pacbot_x as f64,
                    y: pacbot_y as f64,
                },
                angle: pacbot_angle,
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

    pub fn update(&mut self, magnitude: f64, direction: f64, sensors: [f64; 5]) -> ((f64, f64), f64) {
        for point in self.points.iter_mut() {
            point.pos.update_by(magnitude, direction);
            point.angle += direction;
        }

        // sort points by accuracy based on sensors
        let mut point_accuracies: Vec<(&PfPose, f64)> = self.points
            .iter()
            .map(|point| (point, self.get_point_error(point, sensors)))
            .collect();

        point_accuracies.sort_unstable_by_key(|(_, accuracy)| NotNan::new(*accuracy).unwrap());

        let sorted_points: Vec<PfPose> = point_accuracies.into_iter().map(|(point, _)| *point).collect();
        self.points.copy_from_slice(&sorted_points);

        // find the best guess position
        self.pacbot_pose = self.points[0];
        self.update_cell_sort();

        // replace the worst half of the points with random points around the best points
        for i in PARTICLE_FILTER_POINTS / 2..PARTICLE_FILTER_POINTS {
            // choose a random index from 0 to PARTICLE_FILTER_POINTS, weighing lower values more
            let index_range = Uniform::new(0.0, 1.0);
            let mut index_f: f64 = 1.0 - (index_range.sample(&mut rand::thread_rng()) as f64).sqrt();
            index_f *= PARTICLE_FILTER_POINTS as f64 / 2.0;
            let mut index = index_f as usize;
            if index >= PARTICLE_FILTER_POINTS {
                index = PARTICLE_FILTER_POINTS - 1;
            }

            let old_point = self.points[index];
            // choose a random small angle and distance from the old point
            let angle_range = Uniform::new(-0.1, 0.1);
            let x_range = Uniform::new(-0.1, 0.1);
            let y_range = Uniform::new(-0.1, 0.1);

            let angle = old_point.angle + angle_range.sample(&mut rand::thread_rng());
            let x = old_point.pos.x + x_range.sample(&mut rand::thread_rng());
            let y = old_point.pos.y + y_range.sample(&mut rand::thread_rng());

            self.points[i] = PfPose {
                pos: PfPosition { x, y },
                angle,
            };
        }

        // replace some of the new points with completely random points
        for i in (PARTICLE_FILTER_POINTS * 9) / 10..PARTICLE_FILTER_POINTS {
            self.points[i] = self.random_point();
        }

        // return the best guess position
        ((self.pacbot_pose.pos.x, self.pacbot_pose.pos.y), self.pacbot_pose.angle)
    }
}

impl ParticleFilter {
    fn get_point_error(&self, point: &PfPose, sensors: [f64; 5]) -> f64 {
        let mut error = 0.0;

        for i in 0..5 {
            let angle = point.angle + SENSOR_ANGLES[i];
            let distance = self.raycast(point.pos, angle) - SENSOR_DISTANCE_FROM_CENTER;
            let sensor_distance = sensors[i];

            let diff = (distance - sensor_distance).abs();


            error += diff;
        }

        error
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
                let is_wall_here = is_walkable((x, y)) != is_walkable((x, y + 1));
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
                let is_wall_here = is_walkable((x, y)) != is_walkable((x + 1, y));
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

    fn update_cell_sort(&mut self) {
        self.empty_grid_cells.sort_by_key(|a| NotNan::new(a.dist(&self.pacbot_pose.pos)).unwrap());
    }

    fn random_point(&self) -> PfPose {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 10.0).unwrap();
        let random_value = rng.sample::<f64, _>(normal).abs();
        let index = random_value.round() as usize;

        let pos = self.empty_grid_cells[index].clone();

        let extra_space_per_side = (INNER_CELL_WIDTH - ROBOT_WIDTH) / 2.0;

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
}

mod test {
    use super::*;

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < EPSILON);
    }

    #[test]
    fn test_horizontal_segment() {
        let segment = HorizontalSegment {
            x_min: 0.0,
            x_max: 1.0,
            y: 0.0,
        };

        assert_eq!(segment.raycast(0.5, 0.5, 0.0, -1.0), Some(0.5));
        assert_eq!(segment.raycast(0.5, 0.5, 0.0, 1.0), None);
        assert_eq!(segment.raycast(0.5, 0.5, 1.0, 0.0), None);
        assert_eq!(segment.raycast(0.5, 0.5, -1.0, 0.0), None);

        let mut pi_over4 = -std::f64::consts::PI / 4.0;
        assert_close(segment.raycast(0.0, 0.5, pi_over4.cos(), pi_over4.sin()).unwrap(),
                     (2.0 as f64).sqrt() / 2.0);

        pi_over4 *= 3.0;
        assert_eq!(segment.raycast(0.0, 0.5, pi_over4.cos(), pi_over4.sin()), None);

        // from underneath
        assert_eq!(segment.raycast(0.5, -0.5, 0.0, 1.0), Some(0.5));
        assert_eq!(segment.raycast(0.5, -0.5, 0.0, -1.0), None);
        assert_eq!(segment.raycast(0.5, -0.5, 1.0, 0.0), None);
        assert_eq!(segment.raycast(0.5, -0.5, -1.0, 0.0), None);

        pi_over4 = std::f64::consts::PI / 4.0;
        assert_close(segment.raycast(0.0, -0.5, pi_over4.cos(), pi_over4.sin()).unwrap(),
                     (2.0 as f64).sqrt() / 2.0);

        pi_over4 *= 3.0;
        assert_eq!(segment.raycast(0.0, -0.5, pi_over4.cos(), pi_over4.sin()), None);
    }

    #[test]
    fn test_vertical_segment() {
        let segment = VerticalSegment {
            x: 0.0,
            y_min: 0.0,
            y_max: 1.0,
        };

        assert_eq!(segment.raycast(0.5, 0.5, -1.0, 0.0), Some(0.5));
        assert_eq!(segment.raycast(0.5, 0.5, 1.0, 0.0), None);
        assert_eq!(segment.raycast(0.5, 0.5, 0.0, 1.0), None);
        assert_eq!(segment.raycast(0.5, 0.5, 0.0, -1.0), None);

        let mut pi_over4 = std::f64::consts::PI * 3.0 / 4.0;
        assert_close(segment.raycast(0.5, 0.0, pi_over4.cos(), pi_over4.sin()).unwrap(),
                     (2.0 as f64).sqrt() / 2.0);

        pi_over4 = pi_over4 / 3.0 * 5.0;
        assert_eq!(segment.raycast(0.5, 0.0, pi_over4.cos(), pi_over4.sin()), None);

        // from the left
        assert_eq!(segment.raycast(-0.5, 0.5, 1.0, 0.0), Some(0.5));
        assert_eq!(segment.raycast(-0.5, 0.5, -1.0, 0.0), None);
        assert_eq!(segment.raycast(-0.5, 0.5, 0.0, 1.0), None);
        assert_eq!(segment.raycast(-0.5, 0.5, 0.0, -1.0), None);

        pi_over4 = std::f64::consts::PI / 4.0;
        assert_close(segment.raycast(-0.5, 0.0, pi_over4.cos(), pi_over4.sin()).unwrap(),
                     (2.0 as f64).sqrt() / 2.0);

        pi_over4 = pi_over4 * 3.0;
        assert_eq!(segment.raycast(-0.5, 0.0, pi_over4.cos(), pi_over4.sin()), None);
    }

    #[test]
    fn test_pf_position_update_by() {
        let mut pos = PfPosition { x: 0.0, y: 0.0 };
        pos.update_by(1.0, 0.0);
        assert_close(pos.x, 1.0);
        assert_close(pos.y, 0.0);

        pos.update_by(1.0, std::f64::consts::PI / 2.0);
        assert_close(pos.x, 1.0);
        assert_close(pos.y, 1.0);

        pos.update_by(1.0, std::f64::consts::PI);
        assert_close(pos.x, 0.0);
        assert_close(pos.y, 1.0);

        pos.update_by(1.0, std::f64::consts::PI * 3.0 / 2.0);
        assert_close(pos.x, 0.0);
        assert_close(pos.y, 0.0);

        pos.update_by(1.0, std::f64::consts::PI * 2.0);
        assert_close(pos.x, 1.0);
        assert_close(pos.y, 0.0);
    }

    #[test]
    fn test_pf_position_dist() {
        let pos1 = PfPosition { x: 0.0, y: 0.0 };
        let pos2 = PfPosition { x: 1.0, y: 0.0 };
        assert_close(pos1.dist(&pos2), 1.0);

        let pos2 = PfPosition { x: 0.0, y: 1.0 };
        assert_close(pos1.dist(&pos2), 1.0);

        let pos2 = PfPosition { x: 1.0, y: 1.0 };
        assert_close(pos1.dist(&pos2), (2.0 as f64).sqrt());

        let pos2 = PfPosition { x: 2.0, y: 2.0 };
        assert_close(pos1.dist(&pos2), (8.0 as f64).sqrt());
    }
}