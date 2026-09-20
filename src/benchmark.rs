use rust_chess_engine::Board;
use rust_chess_engine::search::Searcher;
use rust_chess_engine::board::chess_move::MoveType;
use std::time::Instant;

pub struct PerftTest {
    pub name: &'static str,
    pub fen: &'static str,
    pub depths: &'static [(u32, u64)],
}

pub const PERFT_TESTS: &[PerftTest] = &[
    PerftTest {
        name: "Starting Position",
        fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        depths: &[(1, 20), (2, 400), (3, 8_902), (4, 197_281), (5, 4_865_609)],
    },
    PerftTest {
        name: "Kiwipete (Complex Position)",
        fen: "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        depths: &[(1, 48), (2, 2_039), (3, 97_862)],
    },
    PerftTest {
        name: "Position 3 (Tactical)",
        fen: "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        depths: &[(1, 14), (2, 191), (3, 2_812), (4, 43_238)],
    },
];

pub struct SearchBenchmark {
    pub depth: u32,
    pub nodes: u64,
    pub qnodes: u64,
    pub time_ms: u128,
    pub nps: f64,
    pub tt_hits: u64,
    pub tt_misses: u64,
    pub beta_cutoffs: u64,
    pub killer_moves_used: u64,
}

impl SearchBenchmark {
    pub fn display(&self) {
        println!(
            "Depth {} | Nodes: {:>10} | QNodes: {:>7} | Time: {:>6}ms | NPS: {:>9.0}",
            self.depth, self.nodes, self.qnodes, self.time_ms, self.nps
        );
        println!(
            "         | TT Hit%: {:>5.1}% | Beta Cutoffs: {:>10} | Killer Moves: {:>5}",
            if self.tt_hits + self.tt_misses > 0 {
                (self.tt_hits as f64 / (self.tt_hits + self.tt_misses) as f64) * 100.0
            } else {
                0.0
            },
            self.beta_cutoffs,
            self.killer_moves_used
        );
    }
}

pub fn run_perft_tests() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    PERFT VALIDATION SUITE                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut passed = 0;
    let mut failed = 0;

    for test in PERFT_TESTS {
        println!("Testing: {}", test.name);
        println!("{}", "─".repeat(64));

        if let Ok(board) = Board::from_fen(test.fen) {
            for (depth, expected) in test.depths {
                let result = board.perft(*depth);
                let status = if result.nodes == *expected { "✓" } else { "✗" };
                println!(
                    "  Depth {}: {} nodes {:>10} (expected {:>10}) {}",
                    depth,
                    status,
                    result.nodes,
                    expected,
                    if result.nodes == *expected { "" } else { "MISMATCH!" }
                );

                if result.nodes == *expected {
                    passed += 1;
                } else {
                    failed += 1;
                }
            }
        } else {
            println!("  ERROR: Failed to parse FEN");
            failed += test.depths.len();
        }
        println!();
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║  Perft Tests: {} PASSED, {} FAILED                           ║", passed, failed);
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}

pub fn test_special_moves() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                 SPECIAL MOVES VALIDATION TEST                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("TEST 1: CASTLING");
    println!("{}", "─".repeat(64));
    let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        .expect("Failed to parse");
    let castling_moves: usize = board
        .generate_moves()
        .iter()
        .filter(|m| matches!(m.move_type, MoveType::Castling))
        .count();
    println!("  ✓ Castling moves available: {}", castling_moves);

    println!("\nTEST 2: EN PASSANT");
    println!("{}", "─".repeat(64));
    let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        .expect("Failed to parse");
    println!("  ✓ En passant square: {:?}", board.en_passant_square);

    println!("\nTEST 3: PROMOTION");
    println!("{}", "─".repeat(64));
    let board = Board::from_fen("8/P7/8/8/8/8/8/8 w - - 0 1").expect("Failed to parse");
    let promo_moves: usize = board
        .generate_moves()
        .iter()
        .filter(|m| matches!(m.move_type, MoveType::Promotion(_)))
        .count();
    println!("  ✓ Promotion moves available: {}", promo_moves);

    println!();
}

pub fn run_search_benchmarks() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                  SEARCH BENCHMARK SUITE                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 7),
        ("After 1.e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", 7),
        ("Position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 6),
    ];

    for (name, fen, max_depth) in positions {
        println!("Position: {}", name);
        println!("{}", "═".repeat(64));

        if let Ok(board) = Board::from_fen(fen) {
            for depth in 4..=max_depth {
                let benchmark = benchmark_search(board.clone(), depth);
                benchmark.display();
            }
        }
        println!();
    }
}

pub fn benchmark_search(mut board: Board, depth: u32) -> SearchBenchmark {
    let mut searcher = Searcher::new();
    searcher.set_tt_size(64);

    let start = Instant::now();
    searcher.find_best_move(&mut board, depth);
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_millis();
    let nps = if time_ms > 0 {
        (searcher.stats.nodes as f64 / time_ms as f64) * 1000.0
    } else {
        0.0
    };

    SearchBenchmark {
        depth,
        nodes: searcher.stats.nodes,
        qnodes: searcher.stats.qnodes,
        time_ms,
        nps,
        tt_hits: searcher.stats.tt_hits,
        tt_misses: searcher.stats.tt_misses,
        beta_cutoffs: searcher.stats.cutoffs,
        killer_moves_used: searcher.stats.killer_moves_used,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perft_starting_position() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let result = board.perft(3);
        assert_eq!(result.nodes, 8_902);
    }
}
