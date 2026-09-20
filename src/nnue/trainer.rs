use ndarray::Array1;
use crate::board::Board;
use super::architecture::*;
use super::network::NNUENetwork;
use super::features::FeatureGenerator;
use super::pgn_loader::{GamePosition, PGNLoader};

pub struct NNUETrainer {
    pub learning_rate: f32,
    pub momentum: f32,
    pub batch_size: usize,
}

pub struct TrainingMetrics {
    pub epoch: usize,
    pub loss: f32,
    pub validation_loss: f32,
    pub accuracy: f32,
}

impl NNUETrainer {
    pub fn new() -> Self {
        NNUETrainer {
            learning_rate: 0.001,
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
    ) -> Result<Vec<TrainingMetrics>, String> {
        println!("Loading PGN dataset from {}...", pgn_path);
        let positions = PGNLoader::load_pgn(pgn_path, Some(50000))?;
        
        if positions.is_empty() {
            return Err("No positions loaded from PGN file".to_string());
        }

        println!("Loaded {} positions", positions.len());

        let split_idx = ((positions.len() as f32) * (1.0 - validation_split)) as usize;
        let (train_data, validation_data) = positions.split_at(split_idx);

        println!("Training set: {} positions", train_data.len());
        println!("Validation set: {} positions", validation_data.len());

        let mut metrics = Vec::new();

        for epoch in 0..epochs {
            println!("\nEpoch {}/{}", epoch + 1, epochs);

            let train_loss = self.train_epoch(network, train_data)?;
            let val_loss = self.validate(network, validation_data)?;
            let accuracy = self.calculate_accuracy(network, validation_data)?;

            metrics.push(TrainingMetrics {
                epoch: epoch + 1,
                loss: train_loss,
                validation_loss: val_loss,
                accuracy,
            });

            println!("  Train Loss: {:.6}", train_loss);
            println!("  Val Loss:   {:.6}", val_loss);
            println!("  Accuracy:   {:.4}", accuracy);
        }

        Ok(metrics)
    }

    fn train_epoch(
        &self,
        network: &mut NNUENetwork,
        data: &[GamePosition],
    ) -> Result<f32, String> {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in data.chunks(self.batch_size) {
            let mut batch_loss = 0.0;

            for position in batch {
                let board = Board::from_fen(&position.fen)
                    .map_err(|e| format!("Invalid FEN: {}", e))?;

                let features = FeatureGenerator::board_to_features(&board);
                let target = position.result;

                let (hidden, activated, output) = network.forward_full(&features);

                let loss = (output - target).powi(2);
                batch_loss += loss;

                let output_error = 2.0 * (output - target);
                self.backprop(network, &features, &hidden, &activated, output_error)?;
            }

            total_loss += batch_loss / (batch.len() as f32);
            batch_count += 1;
        }

        Ok(total_loss / (batch_count as f32))
    }

    fn backprop(
        &self,
        network: &mut NNUENetwork,
        features: &Array1<f32>,
        hidden: &Array1<f32>,
        activated: &Array1<f32>,
        output_error: f32,
    ) -> Result<(), String> {
        let weights = network.get_weights_mut();

        let _output_grad = Array1::from_elem(OUTPUT_SIZE, output_error);

        weights.output_bias -= self.learning_rate * output_error;

        for h in 0..HIDDEN_SIZE {
            weights.output_weights[[0, h]] -= self.learning_rate * output_error * activated[h];
        }

        let mut hidden_error = Array1::zeros(HIDDEN_SIZE);
        for h in 0..HIDDEN_SIZE {
            hidden_error[h] = output_error * weights.output_weights[[0, h]];
        }

        let hidden_deriv: Array1<f32> = hidden.mapv(relu_derivative);

        for h in 0..HIDDEN_SIZE {
            let grad = hidden_deriv[h] * hidden_error[h];
            weights.input_bias[h] -= self.learning_rate * grad;

            for i in 0..INPUT_SIZE {
                weights.input_weights[[h, i]] -= self.learning_rate * grad * features[i];
            }
        }

        Ok(())
    }

    fn validate(
        &self,
        network: &NNUENetwork,
        data: &[GamePosition],
    ) -> Result<f32, String> {
        let mut total_loss = 0.0;

        for position in data {
            let board = Board::from_fen(&position.fen)
                .map_err(|e| format!("Invalid FEN: {}", e))?;

            let features = FeatureGenerator::board_to_features(&board);
            let target = position.result;
            let output = network.forward(&features);

            let loss = (output - target).powi(2);
            total_loss += loss;
        }

        Ok(total_loss / (data.len() as f32))
    }

    fn calculate_accuracy(
        &self,
        network: &NNUENetwork,
        data: &[GamePosition],
    ) -> Result<f32, String> {
        let mut correct = 0;

        for position in data {
            let board = Board::from_fen(&position.fen)
                .map_err(|e| format!("Invalid FEN: {}", e))?;

            let features = FeatureGenerator::board_to_features(&board);
            let target = position.result;
            let output = network.forward(&features);

            let target_class = if target > 0.75 { 1 } else if target < 0.25 { 0 } else { 2 };
            let output_class = if output > 0.75 { 1 } else if output < 0.25 { 0 } else { 2 };

            if target_class == output_class || target_class == 2 || output_class == 2 {
                correct += 1;
            }
        }

        Ok((correct as f32) / (data.len() as f32))
    }
}
