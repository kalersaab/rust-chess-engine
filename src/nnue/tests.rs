/// Comprehensive NNUE test suite
///
/// Tests are grouped into logical sections:
///   1. Architecture constants & weight shapes
///   2. Activation / scaling functions
///   3. Feature generation (full & incremental)
///   4. Forward pass & board evaluation
///   5. Serialization round-trip
///   6. Trainer (single-step gradient descent)

#[cfg(test)]
mod architecture_tests {
    use crate::nnue::architecture::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_constants_are_positive() {
        assert!(INPUT_SIZE > 0, "INPUT_SIZE must be positive");
        assert!(HIDDEN_SIZE > 0, "HIDDEN_SIZE must be positive");
        assert!(OUTPUT_SIZE > 0, "OUTPUT_SIZE must be positive");
        assert!(SCALE_FACTOR > 0.0, "SCALE_FACTOR must be positive");
    }

    #[test]
    fn test_weight_shapes_from_zeros() {
        let w = NNUEWeights::from_zeros();
        assert_eq!(w.input_weights.shape(), &[HIDDEN_SIZE, INPUT_SIZE]);
        assert_eq!(w.input_bias.len(), HIDDEN_SIZE);
        assert_eq!(w.output_weights.shape(), &[OUTPUT_SIZE, HIDDEN_SIZE]);
        assert_eq!(w.output_bias, 0.0);
    }

    #[test]
    fn test_weight_shapes_random_init() {
        let w = NNUEWeights::new();
        assert_eq!(w.input_weights.shape(), &[HIDDEN_SIZE, INPUT_SIZE]);
        assert_eq!(w.input_bias.len(), HIDDEN_SIZE);
        assert_eq!(w.output_weights.shape(), &[OUTPUT_SIZE, HIDDEN_SIZE]);
    }

    #[test]
    fn test_relu_positive() {
        assert_eq!(relu(1.0), 1.0);
        assert_eq!(relu(0.001), 0.001);
    }

    #[test]
    fn test_relu_negative_is_zero() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(-0.001), 0.0);
    }

    #[test]
    fn test_relu_at_zero() {
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_relu_derivative_positive() {
        assert_eq!(relu_derivative(1.0), 1.0);
        assert_eq!(relu_derivative(0.001), 1.0);
    }

    #[test]
    fn test_relu_derivative_non_positive() {
        assert_eq!(relu_derivative(0.0), 0.0);
        assert_eq!(relu_derivative(-1.0), 0.0);
    }

    #[test]
    fn test_output_scaling_clamped_max() {
        let very_large = 1_000_000.0_f32;
        assert_eq!(output_scaling(very_large), 32000);
    }

    #[test]
    fn test_output_scaling_clamped_min() {
        let very_negative = -1_000_000.0_f32;
        assert_eq!(output_scaling(very_negative), -32000);
    }

    #[test]
    fn test_output_scaling_zero() {
        assert_eq!(output_scaling(0.0), 0);
    }

    #[test]
    fn test_output_scaling_roundtrip_approx() {
        let original: i32 = 100;
        let unscaled = output_unscaling(original);
        let rescaled = output_scaling(unscaled);
        assert!((rescaled - original).abs() <= 1, "roundtrip error > 1 cp");
    }

    #[test]
    fn test_nnue_layer_forward() {
        let layer = NNUELayer {
            weights: Array2::eye(3),
            bias: Array1::zeros(3),
        };
        let input = Array1::from(vec![1.0_f32, 2.0, 3.0]);
        let output = layer.forward(&input);
        assert_eq!(output.as_slice().unwrap(), &[1.0_f32, 2.0, 3.0]);
    }

    #[test]
    fn test_nnue_layer_forward_with_activation_clips_negatives() {
        let layer = NNUELayer {
            weights: Array2::from_elem((3, 3), -1.0),
            bias: Array1::zeros(3),
        };
        let input = Array1::from(vec![1.0_f32, 1.0, 1.0]);
        let output = layer.forward_with_activation(&input);
        assert!(output.iter().all(|&x| x == 0.0));
    }
}

#[cfg(test)]
mod feature_tests {
    use crate::board::Board;
    use crate::nnue::architecture::INPUT_SIZE;
    use crate::nnue::features::FeatureGenerator;

    #[test]
    fn test_features_correct_length() {
        let board = Board::new();
        let features = FeatureGenerator::board_to_features(&board);
        assert_eq!(features.len(), INPUT_SIZE);
    }

    #[test]
    fn test_features_binary() {
        let board = Board::new();
        let features = FeatureGenerator::board_to_features(&board);
        for &v in features.iter() {
            assert!(v == 0.0 || v == 1.0, "feature value must be 0 or 1, got {}", v);
        }
    }

    #[test]
    fn test_starting_position_active_features() {
        let board = Board::new();
        let features = FeatureGenerator::board_to_features(&board);
        let active_count = features.iter().filter(|&&v| v == 1.0).count();
        assert_eq!(active_count, 32, "expected 32 active features in start position");
    }

    #[test]
    fn test_piece_type_index_all_variants() {
        use crate::board::pieces::Piece::*;
        assert_eq!(FeatureGenerator::piece_type_index(WhitePawn), 0);
        assert_eq!(FeatureGenerator::piece_type_index(BlackPawn), 0);
        assert_eq!(FeatureGenerator::piece_type_index(WhiteKnight), 1);
        assert_eq!(FeatureGenerator::piece_type_index(BlackKnight), 1);
        assert_eq!(FeatureGenerator::piece_type_index(WhiteBishop), 2);
        assert_eq!(FeatureGenerator::piece_type_index(BlackBishop), 2);
        assert_eq!(FeatureGenerator::piece_type_index(WhiteRook), 3);
        assert_eq!(FeatureGenerator::piece_type_index(BlackRook), 3);
        assert_eq!(FeatureGenerator::piece_type_index(WhiteQueen), 4);
        assert_eq!(FeatureGenerator::piece_type_index(BlackQueen), 4);
        assert_eq!(FeatureGenerator::piece_type_index(WhiteKing), 5);
        assert_eq!(FeatureGenerator::piece_type_index(BlackKing), 5);
    }

    #[test]
    fn test_piece_is_white_correctness() {
        use crate::board::pieces::Piece::*;
        assert!(FeatureGenerator::piece_is_white(WhitePawn));
        assert!(FeatureGenerator::piece_is_white(WhiteKnight));
        assert!(FeatureGenerator::piece_is_white(WhiteBishop));
        assert!(FeatureGenerator::piece_is_white(WhiteRook));
        assert!(FeatureGenerator::piece_is_white(WhiteQueen));
        assert!(FeatureGenerator::piece_is_white(WhiteKing));

        assert!(!FeatureGenerator::piece_is_white(BlackPawn));
        assert!(!FeatureGenerator::piece_is_white(BlackKnight));
        assert!(!FeatureGenerator::piece_is_white(BlackBishop));
        assert!(!FeatureGenerator::piece_is_white(BlackRook));
        assert!(!FeatureGenerator::piece_is_white(BlackQueen));
        assert!(!FeatureGenerator::piece_is_white(BlackKing));
    }

    #[test]
    fn test_feature_index_in_bounds() {
        use crate::board::pieces::Piece::*;
        let pieces = [
            WhitePawn, BlackPawn, WhiteKnight, BlackKnight,
            WhiteBishop, BlackBishop, WhiteRook, BlackRook,
            WhiteQueen, BlackQueen, WhiteKing, BlackKing,
        ];
        for piece in pieces {
            for sq in 0..64 {
                let idx = FeatureGenerator::piece_to_feature(piece, sq);
                assert!(idx < INPUT_SIZE, "feature index {} out of bounds for {:?} sq {}", idx, piece, sq);
            }
        }
    }

    #[test]
    fn test_feature_index_to_square_roundtrip() {
        for sq in 0..64_usize {
            let (rank, file) = FeatureGenerator::feature_index_to_square(sq);
            let recovered = rank * 8 + file;
            assert_eq!(recovered, sq);
        }
    }

    #[test]
    fn test_incremental_features_correct_length() {
        let board = Board::new();
        let full = FeatureGenerator::board_to_features(&board);
        let incremental = FeatureGenerator::board_to_features_incremental(
            &board,
            (7, 7),
            (7, 7),
            &full,
        );
        assert_eq!(incremental.len(), INPUT_SIZE);
    }

    #[test]
    fn test_features_differ_across_positions() {
        let board_start = Board::new();
        let board_after = {
            let mut b = Board::new();
            b.make_move("e2", "e4").ok();
            b
        };
        let f1 = FeatureGenerator::board_to_features(&board_start);
        let f2 = FeatureGenerator::board_to_features(&board_after);
        let changed = f1.iter().zip(f2.iter()).any(|(a, b)| a != b);
        assert!(changed, "features must change after a pawn move");
    }
}

#[cfg(test)]
mod network_tests {
    use ndarray::Array1;
    use crate::board::Board;
    use crate::nnue::architecture::{INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NNUEWeights};
    use crate::nnue::network::NNUENetwork;

    #[test]
    fn test_network_shapes() {
        let net = NNUENetwork::new();
        assert_eq!(net.weights.input_weights.shape(), &[HIDDEN_SIZE, INPUT_SIZE]);
        assert_eq!(net.weights.input_bias.len(), HIDDEN_SIZE);
        assert_eq!(net.weights.output_weights.shape(), &[OUTPUT_SIZE, HIDDEN_SIZE]);
    }

    #[test]
    fn test_zero_input_forward_is_finite() {
        let net = NNUENetwork::new();
        let input = Array1::zeros(INPUT_SIZE);
        let out = net.forward(&input);
        assert!(out.is_finite());
    }

    #[test]
    fn test_ones_input_forward_is_finite() {
        let net = NNUENetwork::new();
        let input = Array1::ones(INPUT_SIZE);
        let out = net.forward(&input);
        assert!(out.is_finite());
    }

    #[test]
    fn test_forward_full_shapes() {
        let net = NNUENetwork::new();
        let input = Array1::zeros(INPUT_SIZE);
        let (hidden, activated, output) = net.forward_full(&input);
        assert_eq!(hidden.len(), HIDDEN_SIZE);
        assert_eq!(activated.len(), HIDDEN_SIZE);
        assert!(output.is_finite());
    }

    #[test]
    fn test_activated_is_non_negative() {
        let net = NNUENetwork::new();
        let input = Array1::zeros(INPUT_SIZE);
        let (_hidden, activated, _output) = net.forward_full(&input);
        assert!(activated.iter().all(|&x| x >= 0.0), "ReLU activations must be non-negative");
    }

    #[test]
    fn test_from_weights_zero_gives_zero_output() {
        let weights = NNUEWeights::from_zeros();
        let net = NNUENetwork::from_weights(weights);
        let input = Array1::zeros(INPUT_SIZE);
        let out = net.forward(&input);
        assert_eq!(out, 0.0);
    }

    #[test]
    fn test_board_evaluation_in_valid_range() {
        let net = NNUENetwork::new();
        let board = Board::new();
        let score = net.evaluate(&board);
        assert!(score >= -32000 && score <= 32000, "score {} out of range", score);
    }

    #[test]
    fn test_evaluation_is_deterministic() {
        let net = NNUENetwork::new();
        let board = Board::new();
        let s1 = net.evaluate(&board);
        let s2 = net.evaluate(&board);
        assert_eq!(s1, s2, "same network + same board must give same score");
    }

    #[test]
    fn test_get_weights_mut_allows_modification() {
        let mut net = NNUENetwork::new();
        {
            let w = net.get_weights_mut();
            w.output_bias = 999.0;
        }
        assert_eq!(net.weights.output_bias, 999.0);
    }

    #[test]
    fn test_get_weights_reference() {
        let net = NNUENetwork::new();
        let w = net.get_weights();
        assert_eq!(w.input_weights.shape(), &[HIDDEN_SIZE, INPUT_SIZE]);
    }
}

#[cfg(test)]
mod serialization_tests {
    use crate::nnue::architecture::{NNUEWeights, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    use crate::nnue::serialization::NNUESerializer;

    fn temp_path(suffix: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("nnue_test_{}.bin", suffix))
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let original = NNUEWeights::new();
        let path = temp_path("roundtrip");

        NNUESerializer::save(&original, &path).expect("save should succeed");
        let loaded = NNUESerializer::load(&path).expect("load should succeed");

        assert_eq!(original.output_bias, loaded.output_bias);
        assert_eq!(original.input_bias[0], loaded.input_bias[0]);
        assert_eq!(original.input_weights[[0, 0]], loaded.input_weights[[0, 0]]);
        assert_eq!(original.output_weights[[0, 0]], loaded.output_weights[[0, 0]]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_zero_weights_roundtrip() {
        let original = NNUEWeights::from_zeros();
        let path = temp_path("zeros");

        NNUESerializer::save(&original, &path).expect("save zeros should succeed");
        let loaded = NNUESerializer::load(&path).expect("load zeros should succeed");

        assert_eq!(loaded.output_bias, 0.0);
        assert!(loaded.input_bias.iter().all(|&x| x == 0.0));
        assert!(loaded.input_weights.iter().all(|&x| x == 0.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_nonexistent_file_returns_error() {
        let result = NNUESerializer::load("/tmp/nonexistent_nnue_1234567890.bin");
        assert!(result.is_err(), "loading non-existent file must return Err");
    }

    #[test]
    fn test_shapes_preserved_after_roundtrip() {
        let original = NNUEWeights::new();
        let path = temp_path("shapes");

        NNUESerializer::save(&original, &path).expect("save");
        let loaded = NNUESerializer::load(&path).expect("load");

        assert_eq!(loaded.input_weights.shape(), &[HIDDEN_SIZE, INPUT_SIZE]);
        assert_eq!(loaded.input_bias.len(), HIDDEN_SIZE);
        assert_eq!(loaded.output_weights.shape(), &[OUTPUT_SIZE, HIDDEN_SIZE]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_all_input_bias_elements_preserved() {
        let original = NNUEWeights::new();
        let path = temp_path("all_elems");

        NNUESerializer::save(&original, &path).expect("save");
        let loaded = NNUESerializer::load(&path).expect("load");

        for i in 0..HIDDEN_SIZE {
            assert_eq!(original.input_bias[i], loaded.input_bias[i],
                "input_bias[{}] mismatch", i);
        }

        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod trainer_tests {
    use crate::board::Board;
    use crate::nnue::network::NNUENetwork;
    use crate::nnue::trainer::NNUETrainer;
    use crate::nnue::features::FeatureGenerator;

    #[test]
    fn test_trainer_default_hyperparams() {
        let trainer = NNUETrainer::new();
        assert!(trainer.learning_rate > 0.0, "lr must be positive");
        assert!(trainer.momentum > 0.0, "momentum must be positive");
        assert!(trainer.batch_size > 0, "batch_size must be positive");
    }

    #[test]
    fn test_single_gradient_step_reduces_loss() {
        let trainer = NNUETrainer::new();
        let mut network = NNUENetwork::new();

        let board = Board::new();
        let features = FeatureGenerator::board_to_features(&board);
        let target = 0.5_f32;

        let initial_output = network.forward(&features);
        let initial_loss = (initial_output - target).powi(2);

        // Manual gradient step on output_bias (the simplest parameter to test)
        let output_error = 2.0 * (initial_output - target);
        {
            let w = network.get_weights_mut();
            w.output_bias -= trainer.learning_rate * output_error;
        }

        let new_output = network.forward(&features);
        let new_loss = (new_output - target).powi(2);

        assert!(
            new_loss <= initial_loss + 1e-6,
            "loss increased after gradient step: {:.6} → {:.6}",
            initial_loss,
            new_loss
        );
    }

    #[test]
    fn test_train_from_pgn_missing_file_returns_error() {
        let trainer = NNUETrainer::new();
        let mut network = NNUENetwork::new();
        let result = trainer.train_from_pgn(
            "/tmp/nonexistent_game_12345.pgn",
            &mut network,
            1,
            0.1,
        );
        assert!(result.is_err(), "training on non-existent file must return Err");
    }
}

#[cfg(test)]
mod sf_probe_tests {
    use crate::nnue::sf_probe::SFNNUEProbe;
    use std::path::Path;

    #[test]
    fn test_probe_nonexistent_file() {
        let result = SFNNUEProbe::probe_file("nonexistent_model_123.nnue");
        assert!(result.is_err());
    }

    #[test]
    fn test_probe_truncated_slice() {
        let dummy = vec![0u8; 8];
        let result = SFNNUEProbe::probe_slice(&dummy);
        assert!(result.is_err());
    }

    #[test]
    fn test_probe_stockfish_net_if_present() {
        let net_path = Path::new("nn-134a887f4c8f.nnue");
        if net_path.exists() {
            let info = SFNNUEProbe::probe_file(net_path).expect("Failed to probe nn-134a887f4c8f.nnue");
            assert!(info.is_valid);
            assert_eq!(info.version, 0x6a448afa);
            assert_eq!(info.architecture_hash, 0xa85b2205);
            assert_eq!(info.feature_transformer_hash, 0xcb685313);
            assert_eq!(info.network_architecture_hash, 0x63337116);
            assert_eq!(info.layer_stacks_count, 8);
            assert_eq!(info.ft_biases_count, 1024);
            assert_eq!(info.ft_sample_biases.len(), 8);
            assert_eq!(info.file_size, 98961994);
            assert_eq!(info.bytes_consumed, 98961994);
        }
    }
}

#[cfg(test)]
mod accumulator_tests {
    use crate::board::Board;
    use crate::nnue::NNUENetwork;

    #[test]
    fn test_accumulator_full_recomputation_matches_evaluate() {
        let network = NNUENetwork::new();
        let board = Board::new();

        let full_eval = network.evaluate(&board);
        let acc = network.create_accumulator(&board);
        let acc_eval = network.evaluate_accumulator(&acc, board.turn);

        assert_eq!(full_eval, acc_eval, "Full evaluation and accumulator evaluation must match");
    }

    #[test]
    fn test_accumulator_incremental_update_matches_recomputation() {
        let network = NNUENetwork::new();
        let mut board = Board::new();
        let mut acc = network.create_accumulator(&board);

        // Sequence of moves: e2e4, e7e5, g1f3, b8c6
        let moves = vec![
            ("e2", "e4"),
            ("e7", "e5"),
            ("g1", "f3"),
            ("b8", "c6"),
        ];

        for (from, to) in moves {
            let moves_list = board.generate_moves();
            let mv = moves_list.iter().find(|m| {
                Board::square_to_string(m.from) == from && Board::square_to_string(m.to) == to
            }).expect("Move should be legal");

            let updated_acc = acc.update_move(&board, mv, &network.weights);
            board.make_move(from, to).expect("make_move should succeed");

            let recomputed_acc = network.create_accumulator(&board);
            let full_eval = network.evaluate(&board);
            let updated_eval = network.evaluate_accumulator(&updated_acc, board.turn);
            let recomputed_eval = network.evaluate_accumulator(&recomputed_acc, board.turn);

            let diff = (updated_eval - recomputed_eval).abs();
            assert!(diff <= 1, "Incremental update evaluation {} differs from recomputed {} (diff={})", updated_eval, recomputed_eval, diff);
            assert_eq!(updated_eval, full_eval, "Incremental evaluation must match full evaluate");

            acc = updated_acc;
        }
    }
}
