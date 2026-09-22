use ndarray::{Array1};
use crate::board::Board;
use crate::evaluation::Score;
use super::architecture::*;
use super::features::FeatureGenerator;

#[derive(Clone)]
pub struct NNUENetwork {
    pub weights: NNUEWeights,
}

impl NNUENetwork {
    pub fn new() -> Self {
        NNUENetwork {
            weights: NNUEWeights::new(),
        }
    }

    pub fn from_weights(weights: NNUEWeights) -> Self {
        NNUENetwork { weights }
    }

    pub fn evaluate(&self, board: &Board) -> Score {
        let features = FeatureGenerator::board_to_features(board);
        let raw_output = self.forward(&features);
        let scaled = output_scaling(raw_output);

        if board.turn == crate::board::Color::White {
            scaled
        } else {
            -scaled
        }
    }

    pub fn create_accumulator(&self, board: &Board) -> super::accumulator::NNUEAccumulator {
        super::accumulator::NNUEAccumulator::compute_from_board(board, &self.weights)
    }

    pub fn evaluate_accumulator(&self, acc: &super::accumulator::NNUEAccumulator, turn: crate::board::Color) -> Score {
        acc.evaluate(&self.weights, turn)
    }

    pub fn forward(&self, features: &Array1<f32>) -> f32 {
        let hidden = self.forward_input_layer(features);
        let activated = hidden.mapv(relu);
        let output = self.forward_output_layer(&activated);

        output[0]
    }

    pub fn forward_input_layer(&self, input: &Array1<f32>) -> Array1<f32> {
        self.weights.input_weights.dot(input) + &self.weights.input_bias
    }

    pub fn forward_output_layer(&self, hidden: &Array1<f32>) -> Array1<f32> {
        self.weights.output_weights.dot(hidden)
            + Array1::from_elem(OUTPUT_SIZE, self.weights.output_bias)
    }

    pub fn forward_full(&self, features: &Array1<f32>) -> (Array1<f32>, Array1<f32>, f32) {
        let hidden = self.forward_input_layer(features);
        let activated = hidden.mapv(relu);
        let output = self.forward_output_layer(&activated);

        (hidden, activated, output[0])
    }

    pub fn get_weights_mut(&mut self) -> &mut NNUEWeights {
        &mut self.weights
    }

    pub fn get_weights(&self) -> &NNUEWeights {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let network = NNUENetwork::new();
        assert_eq!(network.weights.input_weights.shape(), &[HIDDEN_SIZE, INPUT_SIZE]);
        assert_eq!(network.weights.input_bias.len(), HIDDEN_SIZE);
        assert_eq!(network.weights.output_weights.shape(), &[OUTPUT_SIZE, HIDDEN_SIZE]);
    }

    #[test]
    fn test_forward_pass() {
        let network = NNUENetwork::new();
        let input = Array1::zeros(INPUT_SIZE);
        let output = network.forward(&input);
        assert!(output.is_finite());
    }

    #[test]
    fn test_board_evaluation() {
        let network = NNUENetwork::new();
        let board = Board::new();
        let score = network.evaluate(&board);
        assert!(score.abs() < 32000);
    }
}
