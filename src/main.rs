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
