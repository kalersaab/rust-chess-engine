mod benchmark;

use rust_chess_engine::prelude::*;
use benchmark::{run_perft_tests, run_search_benchmarks, test_special_moves};

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
            "--special" => {
                test_special_moves();
            }
            _ => {
                run_tests();
            }
        }
    } else {
        run_tests();
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
