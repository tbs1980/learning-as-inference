extern crate rulinalg;
extern crate rand;

use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;
use rand::distributions::{Normal, IndependentSample};

fn sigmoid(v: f64) ->f64 {
    1.0_f64 / (1.0_f64 + (-v).exp())
}

struct single_neuron_classifier {
    data_x : Matrix<f64>,
    class_t : Vector<f64>
}

// fn log_posterior(w: Vector<f64>){
//
// }
//
// fn grad_log_posterior() {
//
// }

fn main() {
    let num_data_points: usize = 200;
    let num_dim: usize = 2;
    let mut data_x = Matrix::<f64>::zeros(num_data_points, num_dim+1);
    let mut class_t = Vector::<f64>::zeros(num_data_points);

    let normal = Normal::new(0.0, 0.3);

    for ind_i in 0..num_data_points {
        data_x[[ind_i,0]] = 1.0_f64;
        for ind_j in 1..num_dim+1 {
            let v = normal.ind_sample(&mut rand::thread_rng());
            if ind_i < num_data_points / 2 {
                data_x[[ind_i,ind_j]] = v + 1.0_f64;
                class_t[ind_i] = 0.0_f64
            } else {
                data_x[[ind_i,ind_j]] = v + 3.0_f64;
                class_t[ind_i] = 1.0_f64
            }
        }
    }
    let snc = single_neuron_classifier{data_x: data_x, class_t: class_t};
}
