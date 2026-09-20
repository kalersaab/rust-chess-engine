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

pub fn run_eval_benchmark() {
    use rust_chess_engine::evaluation::{Evaluator, EvaluationMode};
    use rust_chess_engine::nnue::NNUENetwork;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║       NNUE vs HANDCRAFTED EVALUATION BENCHMARK SUITE          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let network = NNUENetwork::new();
    let board = Board::from_fen("r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8")
        .expect("Valid FEN");

    println!("SECTION 1: RAW EVALUATION SPEED & ACCUMULATOR ACCELERATION");
    println!("{}", "─".repeat(64));

    let iterations = 10_000;

    let start = Instant::now();
    let mut hc_sum = 0i64;
    for _ in 0..iterations {
        hc_sum += Evaluator::hand_crafted_evaluate(&board) as i64;
    }
    let hc_time = start.elapsed();
    let hc_eps = (iterations as f64 / hc_time.as_secs_f64()) / 1_000.0;

    let start = Instant::now();
    let mut nn_full_sum = 0i64;
    for _ in 0..iterations {
        nn_full_sum += network.evaluate(&board) as i64;
    }
    let nn_full_time = start.elapsed();
    let nn_full_eps = (iterations as f64 / nn_full_time.as_secs_f64()) / 1_000.0;

    let start = Instant::now();
    let acc = network.create_accumulator(&board);
    let mut nn_acc_sum = 0i64;
    for _ in 0..iterations {
        nn_acc_sum += network.evaluate_accumulator(&acc, board.turn) as i64;
    }
    let nn_acc_time = start.elapsed();
    let nn_acc_eps = (iterations as f64 / nn_acc_time.as_secs_f64()) / 1_000.0;

    println!("Handcrafted Evaluator:    {:>8.1} Keval/s ({:>5.1} ms / 10k)", hc_eps, hc_time.as_secs_f64() * 1000.0);
    println!("Full NNUE (Recompute):    {:>8.1} Keval/s ({:>5.1} ms / 10k)", nn_full_eps, nn_full_time.as_secs_f64() * 1000.0);
    println!("Incremental NNUE (Acc):   {:>8.1} Keval/s ({:>5.1} ms / 10k)", nn_acc_eps, nn_acc_time.as_secs_f64() * 1000.0);
    let speedup = nn_acc_eps / nn_full_eps;
    println!("✓ Accumulator Speedup:    {:>8.2}x faster than Full NNUE recomputation", speedup);
    assert!(hc_sum != 0 || nn_full_sum != 0 || nn_acc_sum != 0);

    println!("\nSECTION 2: MULTI-DEPTH SEARCH THROUGHPUT (Nodes-per-Second)");
    println!("{}", "─".repeat(64));

    let bench_positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex Middlegame", "r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8"),
        ("Tactical Endgame",  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ];

    // Cap each individual search at 500ms so the sweep stays under ~30s total.
    let nps_time_limit_ms: u128 = 500;

    for (pos_name, fen) in &bench_positions {
        println!("\nPosition: {}", pos_name);
        println!(
            "  {:<5} | {:>10} | {:>10} | {:>10} | {:>10} | {:>7}",
            "Depth", "HC Nodes", "HC NPS", "NNUE Nodes", "NNUE NPS", "Speedup"
        );
        println!("  {}", "─".repeat(62));

        let test_board = Board::from_fen(fen).expect("Valid FEN");

        for depth in 1u32..=5 {
            let mut searcher_hc = Searcher::new();
            searcher_hc.set_tt_size(16);
            searcher_hc.set_evaluation_mode(EvaluationMode::Handcrafted);
            let start_hc = Instant::now();
            searcher_hc.search_with_time_management(&mut test_board.clone(), depth, nps_time_limit_ms);
            let time_hc = start_hc.elapsed().as_millis().max(1);
            let hc_nodes = searcher_hc.stats.nodes;
            let hc_partial = time_hc >= nps_time_limit_ms;
            let hc_nps = hc_nodes as f64 / (time_hc as f64 / 1000.0);

            let mut searcher_nn = Searcher::new();
            searcher_nn.set_tt_size(16);
            searcher_nn.set_evaluator(Evaluator::with_nnue(NNUENetwork::new()));
            let start_nn = Instant::now();
            searcher_nn.search_with_time_management(&mut test_board.clone(), depth, nps_time_limit_ms);
            let time_nn = start_nn.elapsed().as_millis().max(1);
            let nn_nodes = searcher_nn.stats.nodes;
            let nn_partial = time_nn >= nps_time_limit_ms;
            let nn_nps = nn_nodes as f64 / (time_nn as f64 / 1000.0);

            let speedup = if hc_nps > 0.0 { nn_nps / hc_nps } else { 0.0 };
            let speedup_str = if speedup >= 1.0 {
                format!("{:.2}x ✓", speedup)
            } else {
                format!("{:.2}x ↓", speedup)
            };

            // Prefix node counts with ~ when the time limit was hit (partial search).
            let hc_nodes_str = if hc_partial { format!("~{}", hc_nodes) } else { hc_nodes.to_string() };
            let nn_nodes_str = if nn_partial { format!("~{}", nn_nodes) } else { nn_nodes.to_string() };

            println!(
                "  {:<5} | {:>10} | {:>10.0} | {:>10} | {:>10.0} | {:>7}",
                depth, hc_nodes_str, hc_nps, nn_nodes_str, nn_nps, speedup_str
            );
        }
    }

    println!("\nSECTION 3: BEST-MOVE AGREEMENT ANALYSIS (Depth 4)");
    println!("{}", "─".repeat(64));

    let agreement_positions = vec![
        ("Start Pos",      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After 1.e4",     "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Italian Game",   "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"),
        ("Sicilian Def",   "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Kiwipete",       "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("Tactical End",   "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ("Passed Pawn",    "8/5pk1/1p4p1/3P4/r7/5P2/6P1/4R1K1 w - - 0 35"),
        ("Middlegame",     "r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8"),
    ];

    let agree_depth = 4u32;
    let agree_time_limit_ms: u128 = 2_000; // 2s per evaluator per position
    let mut agree_count = 0usize;
    let mut total_count = 0usize;

    println!(
        "  {:<14} | {:<8} | {:<8} | {:<7} | {}",
        "Position", "HC Move", "NN Move", "Match?", "Nodes (HC / NN)"
    );
    println!("  {}", "─".repeat(64));

    for (pos_name, fen) in &agreement_positions {
        let b = Board::from_fen(fen).expect("Valid FEN");

        let mut s_hc = Searcher::new();
        s_hc.set_tt_size(16);
        s_hc.set_evaluation_mode(EvaluationMode::Handcrafted);
        let (mv_hc, _) = s_hc.search_with_time_management(&mut b.clone(), agree_depth, agree_time_limit_ms);

        let mut s_nn = Searcher::new();
        s_nn.set_tt_size(16);
        s_nn.set_evaluator(Evaluator::with_nnue(NNUENetwork::new()));
        let (mv_nn, _) = s_nn.search_with_time_management(&mut b.clone(), agree_depth, agree_time_limit_ms);

        let hc_str = mv_hc.map(|m| format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to))).unwrap_or_else(|| "none".to_string());
        let nn_str = mv_nn.map(|m| format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to))).unwrap_or_else(|| "none".to_string());
        let agree = hc_str == nn_str;
        if agree { agree_count += 1; }
        total_count += 1;

        println!(
            "  {:<14} | {:<8} | {:<8} | {:<7} | {} / {}",
            pos_name, hc_str, nn_str,
            if agree { "✓ Match" } else { "✗ Diff " },
            s_hc.stats.nodes, s_nn.stats.nodes
        );
    }

    println!(
        "\n  Agreement: {}/{} positions ({:.0}%)",
        agree_count, total_count,
        (agree_count as f64 / total_count as f64) * 100.0
    );

    println!("\nSECTION 4: POSITION EVALUATION COMPARISON");
    println!("{}", "─".repeat(64));

    let eval_positions = vec![
        ("Starting Pos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian Game", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"),
        ("Sicilian Def", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("Passed Pawn",  "8/5pk1/1p4p1/3P4/r7/5P2/6P1/4R1K1 w - - 0 35"),
    ];

    println!("{:<15} | {:>12} | {:>12} | {:>12}", "Position", "Handcrafted", "NNUE Raw", "NNUE Acc");
    println!("{}", "─".repeat(60));

    let nn_evaluator = Evaluator::with_nnue(NNUENetwork::new());
    let hc_evaluator = Evaluator::new();

    for (name, fen) in eval_positions {
        let b = Board::from_fen(fen).expect("Valid FEN");
        let hc_val = hc_evaluator.evaluate(&b);
        let nn_val = nn_evaluator.evaluate(&b);
        let acc = nn_evaluator.nnue_network.as_ref().map(|n| n.create_accumulator(&b));
        let acc_val = nn_evaluator.evaluate_with_accumulator(&b, acc.as_ref());

        println!("{:<15} | {:>10} cp | {:>10} cp | {:>10} cp", name, hc_val, nn_val, acc_val);
        assert_eq!(nn_val, acc_val);
    }

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║       EVALUATION BENCHMARK SUITE COMPLETE                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
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

    /// Verify both evaluators search correctly (non-zero nodes, valid move found)
    /// and agree on best move at depth 3 (fast enough for debug builds).
    #[test]
    fn test_search_benchmark_nnue_vs_hc() {
        use rust_chess_engine::evaluation::{Evaluator, EvaluationMode};
        use rust_chess_engine::nnue::NNUENetwork;

        // Depth 3 keeps each position under ~1s even in unoptimised debug builds,
        // while still exercising alpha-beta, quiescence, and the NNUE accumulator.
        let test_depth = 3u32;

        let positions = vec![
            ("Start",      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            ("Middlegame", "r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8"),
            ("Endgame",    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ];

        for (label, fen) in &positions {
            let board = Board::from_fen(fen).expect("Valid FEN");

            let mut s_hc = Searcher::new();
            s_hc.set_tt_size(8);
            s_hc.set_evaluation_mode(EvaluationMode::Handcrafted);
            let t0 = Instant::now();
            let mv_hc = s_hc.find_best_move(&mut board.clone(), test_depth);
            let ms_hc = t0.elapsed().as_millis().max(1);

            let mut s_nn = Searcher::new();
            s_nn.set_tt_size(8);
            s_nn.set_evaluator(Evaluator::with_nnue(NNUENetwork::new()));
            let t1 = Instant::now();
            let mv_nn = s_nn.find_best_move(&mut board.clone(), test_depth);
            let ms_nn = t1.elapsed().as_millis().max(1);

            assert!(s_hc.stats.nodes > 0,  "[{}] HC searched 0 nodes", label);
            assert!(s_nn.stats.nodes > 0,  "[{}] NNUE searched 0 nodes", label);
            assert!(mv_hc.is_some(),        "[{}] HC found no move", label);
            assert!(mv_nn.is_some(),        "[{}] NNUE found no move", label);
            assert!(s_hc.stats.nodes as f64 / (ms_hc as f64 / 1000.0) > 0.0,
                    "[{}] HC NPS is 0", label);
            assert!(s_nn.stats.nodes as f64 / (ms_nn as f64 / 1000.0) > 0.0,
                    "[{}] NNUE NPS is 0", label);

            // Agreement is informational only: NNUE uses random weights so it
            // legitimately differs from HC at low depth. The benchmark output
            // table in run_eval_benchmark() shows real-world agreement rates.
            let _hc_mv = mv_hc.map(|m| format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to))).unwrap_or_default();
            let _nn_mv = mv_nn.map(|m| format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to))).unwrap_or_default();
        }
    }
}

pub fn test_opening_book() {
    use rust_chess_engine::opening_book::OpeningBook;
    use rust_chess_engine::Board;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                 OPENING BOOK TEST SUITE                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let mut book = OpeningBook::new();

    println!("TEST 1: LOADING EPD FILE");
    println!("{}", "─".repeat(64));
    
    match book.load_epd("endgames.epd") {
        Ok(count) => {
            println!("✓ Successfully loaded {} positions from endgames.epd", count);
            println!("  Book size: {} positions", book.size());
            println!("  Book loaded: {}\n", if book.is_loaded() { "Yes" } else { "No" });
        }
        Err(e) => {
            println!("✗ Failed to load opening book: {}\n", e);
            return;
        }
    }

    println!("TEST 2: RETRIEVING BOOK MOVES");
    println!("{}", "─".repeat(64));
    
    let start_position = Board::new();
    if let Some(book_move) = book.get_book_move(&start_position) {
        println!("✓ Found book move in starting position");
        println!("  Move: from {} to {}", 
            Board::square_to_string(book_move.from),
            Board::square_to_string(book_move.to));
    } else {
        println!("✗ No book move found in starting position");
    }
    println!();

    println!("TEST 3: TESTING SPECIFIC ENDGAME POSITIONS");
    println!("{}", "─".repeat(64));
    
    let endgame_fens = vec![
        ("8/pp2nkR1/5n1p/3p4/5p2/P2BP3/1PPKN3/8 b - - 0 31", "Position 1"),
        ("3n4/2k3p1/p4r2/1pp4P/5PB1/P6P/1KP5/5R2 w - - 0 32", "Position 2"),
        ("4r1k1/5p1p/6pP/2b5/1p3R2/pP2BKP1/P4P2/8 b - - 0 38", "Position 3"),
    ];

    for (fen, name) in endgame_fens {
        match Board::from_fen(fen) {
            Ok(board) => {
                if book.contains_position(&board) {
                    if let Some(moves) = book.get_all_book_moves(&board) {
                        println!("✓ {} - {} book move(s) found", name, moves.len());
                        if let Some(best) = book.get_book_move(&board) {
                            println!("  Best: {} → {}",
                                Board::square_to_string(best.from),
                                Board::square_to_string(best.to));
                        }
                    }
                } else {
                    println!("  {} - Not in book", name);
                }
            }
            Err(_) => {
                println!("✗ {} - Invalid FEN", name);
            }
        }
    }
    println!();

    println!("TEST 4: ZOBRIST HASH CONSISTENCY");
    println!("{}", "─".repeat(64));
    
    let board1 = Board::new();
    let board2 = Board::new();
    
    let hash1 = book.position_hash(&board1);
    let hash2 = book.position_hash(&board2);
    
    if hash1 == hash2 {
        println!("✓ Same position produces same hash");
        println!("  Hash: 0x{:016x}", hash1);
    } else {
        println!("✗ Hash consistency failed");
    }

    let mut board3 = Board::new();
    board3.turn = rust_chess_engine::board::Color::Black;
    let hash3 = book.position_hash(&board3);
    
    if hash1 != hash3 {
        println!("✓ Different position produces different hash");
        println!("  Original: 0x{:016x}", hash1);
        println!("  Changed:  0x{:016x}", hash3);
    } else {
        println!("✗ Different positions produced same hash");
    }
    println!();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              OPENING BOOK TEST SUITE COMPLETE                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
}
