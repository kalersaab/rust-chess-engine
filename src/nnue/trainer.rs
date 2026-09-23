use ndarray::{Array1, Array2};
use crate::board::{Board, Color};
use super::architecture::*;
use super::network::NNUENetwork;
use super::features::FeatureGenerator;
use super::pgn_loader::{GamePosition, PGNLoader};

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

struct VelocityBuffers {
    input_weights: Array2<f32>,
    input_bias: Array1<f32>,
    output_weights: Array2<f32>,
    output_bias: f32,
}

impl VelocityBuffers {
    fn zeros() -> Self {
        VelocityBuffers {
            input_weights: Array2::zeros((HIDDEN_SIZE, INPUT_SIZE)),
            input_bias: Array1::zeros(HIDDEN_SIZE),
            output_weights: Array2::zeros((OUTPUT_SIZE, HIDDEN_SIZE)),
            output_bias: 0.0,
        }
    }
}

pub struct NNUETrainer {
    pub learning_rate: f32,
    pub momentum: f32,
    pub batch_size: usize,
}

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub epoch: usize,
    pub loss: f32,
    pub validation_loss: f32,
    pub accuracy: f32,
}

impl NNUETrainer {
    pub fn new() -> Self {
        NNUETrainer {
            learning_rate: 0.005,
            momentum: 0.9,
            batch_size: 32,
        }
    }

    pub fn train_from_pgn(
        &self,
        pgn_path: &str,
        network: &mut NNUENetwork,
        epochs: usize,
        validation_split: f32,
        max_positions: usize,
        blend_weight: f32,
    ) -> Result<Vec<TrainingMetrics>, String> {
        println!("Loading PGN dataset from {}...", pgn_path);
        let mut positions = PGNLoader::load_pgn(pgn_path, Some(max_positions))?;
        
        if positions.is_empty() {
            return Err("No positions loaded from PGN file".to_string());
        }

        if blend_weight > 0.0 {
            println!(
                "Blending {} labels with handcrafted evaluation (weight {:.2})...",
                positions.len(),
                blend_weight
            );
            for pos in &mut positions {
                if let Ok(board) = Board::from_fen(&pos.fen) {
                    let hc = crate::evaluation::Evaluator::hand_crafted_evaluate(&board);
                    let white_hc = if board.turn == Color::White { hc } else { -hc };
                    let search_prob = sigmoid(white_hc as f32 / 400.0);
                    pos.result = (1.0 - blend_weight) * pos.result + blend_weight * search_prob;
                }
            }
        } else {
            println!(
                "Labels: {} positions with raw game results (blend disabled)",
                positions.len()
            );
        }

        self.train_positions(&positions, network, epochs, validation_split, Some("trained.nnue"))
    }

    pub fn train_positions(
        &self,
        positions: &[GamePosition],
        network: &mut NNUENetwork,
        epochs: usize,
        validation_split: f32,
        save_path: Option<&str>,
    ) -> Result<Vec<TrainingMetrics>, String> {
        if positions.is_empty() {
            return Err("No positions provided for training".to_string());
        }

        let split_idx = ((positions.len() as f32) * (1.0 - validation_split).max(0.1)) as usize;
        let (train_data, validation_data) = positions.split_at(split_idx);

        println!("Dataset: {} total positions", positions.len());
        println!("Training set:   {} positions", train_data.len());
        println!("Validation set: {} positions", validation_data.len());

        let mut velocities = VelocityBuffers::zeros();
        let mut metrics = Vec::new();
        let mut best_val_loss = f32::MAX;

        for epoch in 0..epochs {
            println!("\nEpoch {}/{}", epoch + 1, epochs);

            let train_loss = self.train_epoch(network, train_data, &mut velocities)?;
            let val_loss = self.validate(network, validation_data)?;
            let accuracy = self.calculate_accuracy(network, validation_data)?;

            if val_loss < best_val_loss {
                best_val_loss = val_loss;
                if let Some(path) = save_path {
                    if let Err(e) = super::serialization::NNUESerializer::save(&network.weights, path) {
                        eprintln!("Warning: Failed to save best weights: {}", e);
                    } else {
                        println!("  ★ New best model saved to {}", path);
                    }
                }
            }

            metrics.push(TrainingMetrics {
                epoch: epoch + 1,
                loss: train_loss,
                validation_loss: val_loss,
                accuracy,
            });

            println!("  Train Loss: {:.6}", train_loss);
            println!("  Val Loss:   {:.6}", val_loss);
            println!("  Accuracy:   {:.2}%", accuracy * 100.0);
        }

        Ok(metrics)
    }

    fn train_epoch(
        &self,
        network: &mut NNUENetwork,
        data: &[GamePosition],
        velocities: &mut VelocityBuffers,
    ) -> Result<f32, String> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for position in data {
            let board = match Board::from_fen(&position.fen) {
                Ok(b) => b,
                Err(e) => return Err(format!("Invalid FEN: {}", e)),
            };

            let target = position.result.clamp(0.0, 1.0);

            let (hidden, activated, output) = network.forward_sparse(&board);
            let pred_prob = sigmoid(output);

            let eps = 1e-7_f32;
            let loss = -(target * (pred_prob + eps).ln() + (1.0 - target) * (1.0 - pred_prob + eps).ln());
            total_loss += loss;
            count += 1;

            let output_error = pred_prob - target;
            self.backprop(&board, network, &hidden, &activated, output_error, velocities)?;
        }

        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_loss / (count as f32))
        }
    }

    fn backprop(
        &self,
        board: &Board,
        network: &mut NNUENetwork,
        hidden: &Array1<f32>,
        activated: &Array1<f32>,
        output_error: f32,
        velocities: &mut VelocityBuffers,
    ) -> Result<(), String> {
        let active_indices = FeatureGenerator::active_feature_indices(board);
        let weights = network.get_weights_mut();
        let lr = self.learning_rate;
        let m = self.momentum;

        velocities.output_bias = m * velocities.output_bias - lr * output_error;
        weights.output_bias += velocities.output_bias;

        for h in 0..HIDDEN_SIZE {
            let grad = output_error * activated[h];
            velocities.output_weights[[0, h]] = m * velocities.output_weights[[0, h]] - lr * grad;
            weights.output_weights[[0, h]] += velocities.output_weights[[0, h]];
        }

        let mut hidden_error = Array1::zeros(HIDDEN_SIZE);
        for h in 0..HIDDEN_SIZE {
            hidden_error[h] = output_error * weights.output_weights[[0, h]];
        }

        let hidden_deriv: Array1<f32> = hidden.mapv(relu_derivative);

        for h in 0..HIDDEN_SIZE {
            let grad = hidden_deriv[h] * hidden_error[h];
            if grad.abs() > 1e-8 {
                velocities.input_bias[h] = m * velocities.input_bias[h] - lr * grad;
                weights.input_bias[h] += velocities.input_bias[h];
                for &feat in &active_indices {
                    velocities.input_weights[[h, feat]] =
                        m * velocities.input_weights[[h, feat]] - lr * grad;
                    weights.input_weights[[h, feat]] += velocities.input_weights[[h, feat]];
                }
            }
        }

        Ok(())
    }

    pub fn validate(
        &self,
        network: &NNUENetwork,
        data: &[GamePosition],
    ) -> Result<f32, String> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let mut total_loss = 0.0;
        let eps = 1e-7_f32;

        for position in data {
            let board = match Board::from_fen(&position.fen) {
                Ok(b) => b,
                Err(e) => return Err(format!("Invalid FEN: {}", e)),
            };

            let target = position.result.clamp(0.0, 1.0);
            let output = network.forward_sparse(&board).2;
            let pred_prob = sigmoid(output);

            let loss = -(target * (pred_prob + eps).ln() + (1.0 - target) * (1.0 - pred_prob + eps).ln());
            total_loss += loss;
        }

        Ok(total_loss / (data.len() as f32))
    }

    pub fn calculate_accuracy(
        &self,
        network: &NNUENetwork,
        data: &[GamePosition],
    ) -> Result<f32, String> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let mut correct = 0;

        for position in data {
            let board = match Board::from_fen(&position.fen) {
                Ok(b) => b,
                Err(e) => return Err(format!("Invalid FEN: {}", e)),
            };

            let target = position.result;
            let output = network.forward_sparse(&board).2;
            let pred_prob = sigmoid(output);

            let target_white_favored = target > 0.55;
            let target_black_favored = target < 0.45;
            let pred_white_favored = pred_prob > 0.55;
            let pred_black_favored = pred_prob < 0.45;

            if (target_white_favored && pred_white_favored)
                || (target_black_favored && pred_black_favored)
                || (!target_white_favored && !target_black_favored && !pred_white_favored && !pred_black_favored)
            {
                correct += 1;
            }
        }

        Ok((correct as f32) / (data.len() as f32))
    }
}
