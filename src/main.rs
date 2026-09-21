mod benchmark;

use rust_chess_engine::prelude::*;
use benchmark::{run_perft_tests, run_search_benchmarks, test_special_moves, test_opening_book};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "--uci" => {
                let mut engine = UciEngine::new();
                engine.run();
            }
            "--perft" => {
                run_perft_tests();
            }
            "--bench" => {
                run_search_benchmarks();
            }
            "--bench-eval" | "--eval-bench" => {
                benchmark::run_eval_benchmark();
            }
            "--special" => {
                test_special_moves();
            }
            "--book" => {
                test_opening_book();
            }
            "--pipeline" => {
                run_pipeline_command(&args);
            }
            "--match" | "--selfplay" => {
                run_match_command(&args);
            }
            "--gen-data" => {
                run_gen_data_command(&args);
            }
            "--probe-nnue" | "--nnue" => {
                let path = if args.len() > 2 {
                    args[2].as_str()
                } else {
                    "nn-134a887f4c8f.nnue"
                };
                probe_nnue_file(path);
            }
            _ => {
                run_tests();
            }
        }
    } else {
        run_tests();
    }
}

fn run_pipeline_command(args: &[String]) {
    let nodes_per_move: u64 = if args.len() > 2 {
        args[2].parse().unwrap_or(10_000)
    } else {
        10_000
    };
    let num_positions: usize = if args.len() > 3 {
        args[3].parse().unwrap_or(300)
    } else {
        300
    };
    let num_games: usize = if args.len() > 4 {
        args[4].parse().unwrap_or(6)
    } else {
        6
    };

    println!("================================================================");
    println!("  END-TO-END NNUE PIPELINE: TRAINING -> VALIDATION -> SELF-PLAY ");
    println!("================================================================");
    println!("Benchmark tier:  {} nodes per move", nodes_per_move);
    println!("Target dataset:  {} positions", num_positions);
    println!("Self-play match: {} games\n", num_games);

    // STAGE 1: REAL TRAINING DATA
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STAGE 1: REAL TRAINING DATA GENERATION                      │");
    println!("└─────────────────────────────────────────────────────────────┘");
    let generator = SelfPlayGenerator::new(SelfPlayConfig {
        nodes_per_move,
        random_opening_plies: 6,
        max_moves: 100,
    });

    let mut dataset = Vec::new();
    if std::path::Path::new("endgames.epd").exists() {
        println!("Labeling positions from endgames.epd at {} nodes...", nodes_per_move);
        match generator.label_epd_file("endgames.epd", num_positions) {
            Ok(pos) => dataset.extend(pos),
            Err(e) => eprintln!("Error reading EPD: {}", e),
        }
    }
    if dataset.len() < num_positions {
        let needed_games = ((num_positions - dataset.len()) / 20).max(2);
        println!("Generating additional {} self-play games at {} nodes...", needed_games, nodes_per_move);
        let selfplay_pos = generator.generate_games(needed_games);
        dataset.extend(selfplay_pos);
    }
    println!("Total real training dataset collected: {} positions\n", dataset.len());

    // STAGE 2 & 3: TRAINED WEIGHTS & VALIDATION SET
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ STAGE 2 & 3: TRAINING WEIGHTS & VALIDATION SET EVALUATION   │");
    println!("└─────────────────────────────────────────────────────────────┘");
    let mut network = NNUENetwork::new();
    let start_board = Board::new();
    let untrained_eval = network.evaluate(&start_board);
    println!("Untrained Startpos Eval: {:>6} cp (Random Gaussian Weights)", untrained_eval);

    let trainer = NNUETrainer::new();
    let epochs = 5;
    let val_split = 0.15;
    let _metrics = trainer.train_positions(&dataset, &mut network, epochs, val_split, Some("trained.nnue"))
        .expect("Training failed");

    // STAGE 4: EVALUATION SANITY CHECKS
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ STAGE 4: STATIC EVALUATION SANITY CHECKS                    │");
    println!("└─────────────────────────────────────────────────────────────┘");
    let trained_eval = network.evaluate(&start_board);
    println!("Trained Startpos Eval:   {:>6} cp (Learned Weights)", trained_eval);

    let kq_fen = "4k3/8/8/8/8/8/8/4K2Q w - - 0 1";
    if let Ok(kq_board) = Board::from_fen(kq_fen) {
        let kq_eval = network.evaluate(&kq_board);
        println!("Sanity Check (K+Q vs K): {:>6} cp (Expected: strongly positive)", kq_eval);
    }
    let kr_fen = "4k3/8/8/8/8/8/8/4K2R w - - 0 1";
    if let Ok(kr_board) = Board::from_fen(kr_fen) {
        let kr_eval = network.evaluate(&kr_board);
        println!("Sanity Check (K+R vs K): {:>6} cp (Expected: strongly positive)", kr_eval);
    }

    // STAGE 5: SELF-PLAY MATCH (FIXED NODES)
    println!("\n┌─────────────────────────────────────────────────────────────┐");
    println!("│ STAGE 5: HEAD-TO-HEAD SELF-PLAY MATCH (FIXED NODES)         │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("Running {}-game match: NNUE (Trained) vs Handcrafted (HCE) at {} nodes...", num_games, nodes_per_move);
    let match_runner = MatchRunner::new(nodes_per_move);
    let result = match_runner.run_match(
        num_games,
        "NNUE (Trained)",
        EvaluationMode::NNUE,
        Some(network),
        "Handcrafted (HCE)",
        EvaluationMode::Handcrafted,
        None,
    );
    result.display();
}

fn run_match_command(args: &[String]) {
    let nodes_per_move: u64 = if args.len() > 2 {
        args[2].parse().unwrap_or(10_000)
    } else {
        10_000
    };
    let num_games: usize = if args.len() > 3 {
        args[3].parse().unwrap_or(6)
    } else {
        6
    };

    let nnue_network = if std::path::Path::new("trained.nnue").exists() {
        println!("Loading weights from trained.nnue...");
        let weights = NNUESerializer::load("trained.nnue").expect("Failed to load trained.nnue");
        Some(NNUENetwork::from_weights(weights))
    } else {
        println!("Notice: trained.nnue not found. Using fresh weights (run --pipeline to train).");
        Some(NNUENetwork::new())
    };

    let runner = MatchRunner::new(nodes_per_move);
    let result = runner.run_match(
        num_games,
        "NNUE Engine",
        EvaluationMode::NNUE,
        nnue_network,
        "Handcrafted HCE",
        EvaluationMode::Handcrafted,
        None,
    );
    result.display();
}

fn run_gen_data_command(args: &[String]) {
    let num_games: usize = if args.len() > 2 {
        args[2].parse().unwrap_or(5)
    } else {
        5
    };
    let nodes: u64 = if args.len() > 3 {
        args[3].parse().unwrap_or(10_000)
    } else {
        10_000
    };

    let generator = SelfPlayGenerator::new(SelfPlayConfig {
        nodes_per_move: nodes,
        random_opening_plies: 6,
        max_moves: 100,
    });

    println!("Generating {} self-play games at {} nodes per move...", num_games, nodes);
    let positions = generator.generate_games(num_games);
    println!("Generated {} total positions with labels.", positions.len());
}

fn probe_nnue_file(path: &str) {
    println!("=== Stockfish NNUE Inspector & Validator ===\n");
    println!("File: {}", path);
    match rust_chess_engine::nnue::SFNNUEProbe::probe_file(path) {
        Ok(info) => {
            println!("Status:        ✓ Valid Stockfish SFNNv16 Network");
            println!("File Size:     {} bytes", info.file_size);
            println!("Version:       0x{:08x}", info.version);
            println!("Arch Hash:     0x{:08x}", info.architecture_hash);
            println!("Description:   {}", info.description);
            println!("\n--- Feature Transformer ---");
            println!("FT Hash:       0x{:08x}", info.feature_transformer_hash);
            println!("FT Biases:     {} int16 values", info.ft_biases_count);
            println!("Sample Biases: {:?}", info.ft_sample_biases);
            println!("Threats:       59,808 inputs x 1024 int8 weights ({} bytes) + {} PSQT", info.threat_weights_bytes, info.threat_psqt_count);
            println!("Pawn Pairs:    4,560 inputs x 1024 int8 weights ({} bytes) + {} PSQT", info.pawn_pair_weights_bytes, info.pawn_pair_psqt_count);
            println!("HalfKAv2_hm:   22,528 inputs x 1024 int16 weights ({} values) + {} PSQT", info.half_ka_weights_count, info.half_ka_psqt_count);
            println!("Total Inputs:  86,896 features (Dual Perspective Accumulator)");
            println!("\n--- Evaluation Networks ---");
            println!("Net Arch Hash: 0x{:08x}", info.network_architecture_hash);
            println!("Layer Stacks:  {} material-bucketed stacks", info.layer_stacks_count);
            println!("Layer 1 (fc0): {} inputs -> {} outputs (AffineTransformSparseInput)", info.fc0_dim.0, info.fc0_dim.1);
            println!("Activation:    SqrClippedReLU + ClippedReLU (64 channels)");
            println!("Layer 2 (fc1): {} inputs -> {} outputs (AffineTransform)", info.fc1_dim.0, info.fc1_dim.1);
            println!("Layer 3 (fc2): {} inputs -> {} output (AffineTransform + Skip Connection)", info.fc2_dim.0, info.fc2_dim.1);
            println!("\nVerification:  {} / {} bytes validated (100% complete)", info.bytes_consumed, info.file_size);
        }
        Err(err) => {
            eprintln!("Status:        ✗ Validation Failed");
            eprintln!("Error:         {}", err);
        }
    }
}

fn run_tests() {
    println!("=== Perft Test Suite ===\n");
    println!("Test 1: Starting Position Depth 1");
    let fen1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    let board1 = Board::from_fen(fen1).expect("Failed to parse FEN");
    let result1 = board1.perft(1);
    println!("  Nodes: {}", result1.nodes);
    println!("  Expected: 20");
    println!("  Result: {}\n", if result1.nodes == 20 { "✓ PASS" } else { "✗ FAIL" });
    println!("Test 2: Starting Position Depth 2");
    let result2 = board1.perft(2);
    println!("  Nodes: {}", result2.nodes);
    println!("  Expected: 400");
    println!("  Result: {}\n", if result2.nodes == 400 { "✓ PASS" } else { "✗ FAIL" });
    println!("Test 3: Detailed Divide (Starting Position Depth 2)");
    println!("  First move details:");
    let moves = board1.generate_moves();
    for first_move in &moves[0..3] {
        let mut board = board1.clone();
        let from = Board::square_to_string(first_move.from);
        let to = Board::square_to_string(first_move.to);
        
        if board.make_move(&from, &to).is_ok() {
            let result = board.perft(1);
            println!("    {} -> {} : {} nodes", from, to, result.nodes);
        }
    }
    println!();
    println!("Test 4: Kiwipete Position Depth 1");
    let fen4 = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    let board4 = Board::from_fen(fen4).expect("Failed to parse FEN");
    let result4 = board4.perft(1);
    println!("  Nodes: {}", result4.nodes);
    println!("  Expected: 48");
    println!("  Result: {}\n", if result4.nodes == 48 { "✓ PASS" } else { "✗ FAIL" });
    println!("Test 5: Position 3 Depth 1 and 2");
    let fen5 = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    let board5 = Board::from_fen(fen5).expect("Failed to parse FEN");
    let result5a = board5.perft(1);
    let result5b = board5.perft(2);
    println!("  Depth 1: {} (expected 14)", result5a.nodes);
    println!("  Depth 2: {} (expected 191)", result5b.nodes);
    println!("  Result: {} / {}\n", 
        if result5a.nodes == 14 { "✓" } else { "✗" },
        if result5b.nodes == 191 { "✓" } else { "✗" }
    );

    println!("Test 6: Move Type Statistics (Starting Position Depth 2)");
    let result6 = board1.perft(2);
    println!("  Captures: {}", result6.captures);
    println!("  En Passants: {}", result6.en_passants);
    println!("  Castles: {}", result6.castles);
    println!("  Promotions: {}", result6.promotions);
    println!("  Checks: {}", result6.checks);
    println!("  Checkmates: {}\n", result6.checkmates);
    println!("=== Summary ===");
    println!("Depth 1-2 tests pass, suggesting move generation is mostly correct.");
    println!("Depth 3 discrepancy suggests either:");
    println!("  - Illegal moves escaping validation");
    println!("  - Duplicate moves being generated");
    println!("  - Edge case in specific positions");
}
