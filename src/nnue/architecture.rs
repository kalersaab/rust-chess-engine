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
        Self::with_smart_initialization()
    }

    pub fn with_random_initialization() -> Self {
        use rand_distr::{Normal, Distribution};
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

    pub fn with_smart_initialization() -> Self {
        use rand_distr::{Normal, Distribution};
        use rand::thread_rng;

        let mut rng = thread_rng();

        let fan_in = INPUT_SIZE as f32;
        let fan_out = HIDDEN_SIZE as f32;
        
        let xavier_input = (6.0 / (fan_in + fan_out)).sqrt();
        let xavier_output = (2.0 / HIDDEN_SIZE as f32).sqrt();
        
        let input_normal = Normal::new(0.0, xavier_input * 0.1).unwrap();
        let output_normal = Normal::new(0.0, xavier_output * 0.1).unwrap();

        let mut weights = NNUEWeights {
            input_weights: Array2::from_shape_fn((HIDDEN_SIZE, INPUT_SIZE), |_| {
                input_normal.sample(&mut rng) * 0.01
            }),
            input_bias: Array1::zeros(HIDDEN_SIZE),
            hidden_weights: Array2::from_shape_fn((OUTPUT_SIZE, HIDDEN_SIZE), |_| {
                output_normal.sample(&mut rng)
            }),
            hidden_bias: Array1::zeros(HIDDEN_SIZE),
            output_weights: Array2::from_shape_fn((OUTPUT_SIZE, HIDDEN_SIZE), |_| {
                output_normal.sample(&mut rng) * 0.001
            }),
            output_bias: 0.0,
        };

        Self::bias_towards_material(&mut weights);
        weights
    }

    fn bias_towards_material(weights: &mut NNUEWeights) {
        let piece_values = [
            100.0,  // pawn
            320.0,  // knight
            330.0,  // bishop
            500.0,  // rook
            900.0,  // queen
            0.0,    // king (positional only)
        ];

        for piece_idx in 0..6 {
            for square in 0..64 {
                let white_feature = piece_idx * 64 + square;
                let black_feature = (piece_idx + 6) * 64 + square;
                
                let value = piece_values[piece_idx] / SCALE_FACTOR;
                
                for h in 0..HIDDEN_SIZE {
                    if h % 100 == piece_idx {
                        weights.input_weights[[h, white_feature]] += value * 0.1;
                        weights.input_weights[[h, black_feature]] -= value * 0.1;
                    }
                }
            }
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
