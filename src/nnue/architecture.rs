use ndarray::{Array1, Array2};

pub const INPUT_SIZE: usize = 768;
pub const HIDDEN_SIZE: usize = 32768;
pub const OUTPUT_SIZE: usize = 1;

pub const SCALE_FACTOR: f32 = 361.0;
pub const QA: i32 = 255;
pub const QB: i32 = 64;

#[derive(Clone)]
pub struct NNUEWeights {
    pub input_weights: Array2<f32>,
    pub input_bias: Array1<f32>,
    pub hidden_weights: Array2<f32>,
    pub hidden_bias: Array1<f32>,
    pub output_weights: Array2<f32>,
    pub output_bias: f32,
}

impl NNUEWeights {
    pub fn new() -> Self {
        use rand_distr::Normal;
        use rand_distr::Distribution;
        use rand::thread_rng;

        let mut rng = thread_rng();

        let input_normal = Normal::new(0.0, 0.2).unwrap();
        let output_normal = Normal::new(0.0, 0.1).unwrap();

        NNUEWeights {
            input_weights: Array2::from_shape_fn((HIDDEN_SIZE, INPUT_SIZE), |_| {
                input_normal.sample(&mut rng)
            }),
            input_bias: Array1::from_shape_fn(HIDDEN_SIZE, |_| input_normal.sample(&mut rng)),
            hidden_weights: Array2::from_shape_fn((OUTPUT_SIZE, HIDDEN_SIZE), |_| {
                output_normal.sample(&mut rng)
            }),
            hidden_bias: Array1::from_shape_fn(HIDDEN_SIZE, |_| output_normal.sample(&mut rng)),
            output_weights: Array2::from_shape_fn((OUTPUT_SIZE, HIDDEN_SIZE), |_| {
                output_normal.sample(&mut rng)
            }),
            output_bias: output_normal.sample(&mut rng),
        }
    }

    pub fn from_zeros() -> Self {
        NNUEWeights {
            input_weights: Array2::zeros((HIDDEN_SIZE, INPUT_SIZE)),
            input_bias: Array1::zeros(HIDDEN_SIZE),
            hidden_weights: Array2::zeros((OUTPUT_SIZE, HIDDEN_SIZE)),
            hidden_bias: Array1::zeros(HIDDEN_SIZE),
            output_weights: Array2::zeros((OUTPUT_SIZE, HIDDEN_SIZE)),
            output_bias: 0.0,
        }
    }
}

pub struct NNUELayer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

impl NNUELayer {
    pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(input) + &self.bias
    }

    pub fn forward_with_activation(&self, input: &Array1<f32>) -> Array1<f32> {
        let output = self.forward(input);
        output.mapv(|x| x.max(0.0))
    }
}

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

pub fn output_scaling(raw_score: f32) -> i32 {
    ((raw_score * SCALE_FACTOR) as i32).max(-32000).min(32000)
}

pub fn output_unscaling(score: i32) -> f32 {
    score as f32 / SCALE_FACTOR
}
