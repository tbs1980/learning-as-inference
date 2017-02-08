extern crate rand;
extern crate csv;

use std::path::Path;
use rand::distributions::{Normal, Range, IndependentSample};
use csv::Writer;

fn sigmoid(v: f64) ->f64 {
    1.0_f64 / (1.0_f64 + (-v).exp())
}

struct SingleNeuronClassifier {
    x: Vec<Vec<f64>>,
    t: Vec<f64>
}

impl SingleNeuronClassifier {
    fn log_posterior(&self, w: &Vec<f64>) -> f64{
        assert_eq!(w.len(), self.x[0].len());
        let num_params = w.len();
        let num_data_points = self.x.len();
        let mut g_w: f64 = 0_f64;
        for i in 0..num_data_points {
            let ref x_row = self.x[i];
            let mut a: f64 = 0_f64;
            for j in 0..num_params {
                a += x_row[j]*w[j];
            }
            let y: f64 = sigmoid(a);
            assert!(y > 0_f64);
            assert!(y < 1_f64);
            g_w -= self.t[i]*(y).ln() + ( 1_f64 - self.t[i] )*(1_f64 - y).ln();
        }
        let mut e_w = 0_f64;
        for j in 0..num_params {
            e_w += w[j]*w[j];
        }
        e_w *= 0.5_f64;
        let alpha = 1_f64;
        return g_w + alpha*e_w;
    }

    fn grad_log_posterior(&self, w: &Vec<f64>) -> Vec<f64> {
        assert_eq!(w.len(), self.x[0].len());
        let num_params = w.len();
        let num_data_points = self.x.len();
        let mut g_w = vec![0_f64; num_params];
        for i in 0..num_data_points {
            let ref x_row = self.x[i];
            let mut a: f64 = 0_f64;
            for j in 0..num_params {
                a += x_row[j]*w[j];
            }
            let y: f64 = sigmoid(a);
            let err = self.t[i] - y;
            for j in 0..num_params {
                g_w[j] -= err*self.x[i][j];
            }
        }
        let alpha = 1_f64;
        for j in 0..num_params {
            g_w[j] += w[j]*alpha;
        }
        return g_w;
    }

    fn generate_samples_with_hmc(&self, num_samples: usize) -> Vec<Vec<f64>> {
        let num_params = self.x[0].len();;
        let normal = Normal::new(0_f64, 1_f64);
        let uniform = Range::new(0_f64, 1_f64);
        let mut samps_acc: usize = 0;
        let mut tot_samps: usize = 0;
        let mut w = vec![0_f64; num_params];
        for j in 0..num_params {
            w[j] = uniform.ind_sample(&mut rand::thread_rng());
        }
        let num_steps: usize = 10;
        let epsilon: f64 = 1e-3;
        let mut samples: Vec<Vec<f64>> = Vec::new();
        while samps_acc < num_samples {
            let mut p = vec![0_f64; num_params];
            for j in 0..num_params {
                p[j] = normal.ind_sample(&mut rand::thread_rng());
            }
            let mut pot_eng: f64 = self.log_posterior(&w);
            let mut kin_eng: f64 = 0_f64;
            for j in 0..num_params {
                kin_eng += p[j]*p[j];
            }
            let hamilt = kin_eng + pot_eng;
            let mut w_new = vec![0_f64; num_params];
            for j in 0..num_params {
                w_new[j] = w[j];
            }

            let mut g_new = self.grad_log_posterior(&w_new);
            for _ in 0..num_steps {
                for j in 0..num_params {
                    p[j] = p[j] - 0.5*epsilon*g_new[j];
                    w_new[j] = w_new[j] + epsilon*p[j]
                }
                g_new = self.grad_log_posterior(&w_new);
                for j in 0..num_params {
                    p[j] = p[j] - 0.5*epsilon*g_new[j];
                }
            }
            pot_eng = self.log_posterior(&w_new);
            kin_eng = 0_f64;
            for j in 0..num_params {
                kin_eng += p[j]*p[j];
            }
            let hamilt_new: f64 = kin_eng + pot_eng;
            let delta_h: f64 = hamilt_new - hamilt;
            // println!("delta_h = {}", delta_h);

            let rand_uni: f64 = uniform.ind_sample(&mut rand::thread_rng());
            // println!("rand_uni = {}", rand_uni);
            if delta_h < 0_f64 || rand_uni < (-delta_h).exp(){
                for j in 0..num_params {
                    w[j] = w_new[j];
                }
                samps_acc += 1;
                samples.push(w_new);
            }
            tot_samps +=1;
        }
        let acc_rate: f64 = (samps_acc as f64) / (tot_samps as f64);
        println!("acceptance rate = {} ",  acc_rate);
        return samples;
    }
}

fn main() {
    let num_data_points: usize = 200;
    let num_dims: usize = 2;
    let mut x = vec![vec![0_f64; num_dims+1]; num_data_points];
    let mut t = vec![0_f64; num_data_points];

    let normal = Normal::new(0.0_f64, 1.0_f64);

    for i in 0..num_data_points {
        x[i][0] = 1.0_f64;
        for j in 1..num_dims+1 {
            let v = normal.ind_sample(&mut rand::thread_rng());
            if i < num_data_points / 2 {
                x[i][j] = v + 1.0_f64;
                t[i] = 0.0_f64
            } else {
                x[i][j] = v + 3.0_f64;
                t[i] = 1.0_f64
            }
        }
    }
    let snc = SingleNeuronClassifier{x: x, t: t};

    let num_samples: usize = 10000;
    let samples = snc.generate_samples_with_hmc(num_samples);

    let path = Path::new("samples.csv");
    let mut writer = Writer::from_file(path).unwrap();
    for row in samples.into_iter() {
        writer.encode(row).expect("CSV writer error");
    }
}
