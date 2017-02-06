extern crate rulinalg;
extern crate rand;

use rulinalg::vector::Vector;
use rulinalg::matrix::Matrix;
use rulinalg::matrix::BaseMatrix;
use rand::distributions::{Normal, IndependentSample};

fn sigmoid(v: f64) ->f64 {
    1.0_f64 / (1.0_f64 + (-v).exp())
}

struct SingleNeuronClassifier {
    data_x : Matrix<f64>,
    class_t : Vector<f64>
}

impl SingleNeuronClassifier {
    fn log_posterior(&self, w: Vector<f64>) -> f64{
        assert_eq!(w.size(), self.data_x.cols());
        let mut g_w: f64 = 0_f64;
        for ind_i in 0..self.data_x.rows() {
            let data_x_row = self.data_x.row(ind_i);
            let data_x_row_1 : Vector<f64> = data_x_row.into();
            let a = data_x_row_1.dot(&w);
            let y: f64 = sigmoid(a);
            g_w -= self.class_t[ind_i]*(y).ln() + ( 1_f64 - self.class_t[ind_i]*(1_f64 - y).ln() );
        }
        let mut e_w = 0_f64;
        for ind_j in 0..w.size() {
            e_w += w[ind_j]*w[ind_j];
        }
        e_w *= 0.5;
        let alpha = 1_f64;
        return g_w + alpha*e_w;
    }
}

impl SingleNeuronClassifier {
    fn grad_log_posterior(&self, w: Vector<f64>) -> Vector<f64> {
        assert_eq!(w.size(), self.data_x.cols());
        let mut grad_w = Vector::<f64>::zeros(w.size());
        for ind_i in 0..self.data_x.rows() {
            let data_x_row = self.data_x.row(ind_i);
            let data_x_row_1 : Vector<f64> = data_x_row.into();
            let a = data_x_row_1.dot(&w);
            let y: f64 = sigmoid(a);
            let err = self.class_t[ind_i] - y;
            for ind_j in 0..w.size() {
                grad_w[ind_j] -= err*self.data_x[[ind_i, ind_j]];
            }
        }
        let alpha = 1_f64;
        grad_w += w*alpha;
        return grad_w;
    }
}

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
    let snc = SingleNeuronClassifier{data_x: data_x, class_t: class_t};

    let weights_w = Vector::<f64>::zeros(num_dim+1);
    let log_posterior_val = snc.log_posterior(weights_w);
    println!("the log posterior value is {}", log_posterior_val);

    let weights_w_1 = Vector::<f64>::zeros(num_dim+1);
    let grad_log_posterior_val = snc.grad_log_posterior(weights_w_1);
    println!("the gradient of the log posterior is {}", grad_log_posterior_val);
}
