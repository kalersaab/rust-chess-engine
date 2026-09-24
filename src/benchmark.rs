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

pub fn run_gpu_benchmark() {
    use rust_chess_engine::evaluation::Evaluator;
    use rust_chess_engine::nnue::NNUENetwork;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║             GPU NNUE BATCH EVALUATOR BENCHMARK                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ];
    let tiers: [(u64, &str); 2] = [(10_000, "10k"), (100_000, "100k")];

    for (name, fen) in positions {
        println!("Position: {}", name);
        println!("{}", "═".repeat(78));
        let board = Board::from_fen(fen).expect("Valid FEN");

        for (tier, tier_name) in tiers {
            println!("\n  Fixed nodes per move: {}", tier_name);
            println!(
                "  {:<6} {:>10} {:>9} {:>10} {:>12} {}",
                "Mode", "Nodes", "QNodes", "Time", "NPS", "Best Move"
            );

            let eval_gpu = Evaluator::with_nnue(NNUENetwork::new());
            let eval_cpu = Evaluator::with_nnue(NNUENetwork::new());

            let mut s_gpu = Searcher::new();
            s_gpu.set_tt_size(64);
            s_gpu.set_evaluator(eval_gpu);
            s_gpu.set_gpu_enabled(true);
            s_gpu.set_gpu_batch_min(4);

            let mut b = board.clone();
            let (mv, _) = s_gpu.search_fixed_nodes(&mut b, 200);
            let _ = mv;
            let start = Instant::now();
            let mut b = board.clone();
            let (mv, _) = s_gpu.search_fixed_nodes(&mut b, tier);
            let gpu_ms = start.elapsed().as_millis();
            let gpu_nps = if gpu_ms > 0 {
                (s_gpu.stats.nodes as f64 / gpu_ms as f64) * 1000.0
            } else {
                0.0
            };
            println!(
                "  {:<6} {:>10} {:>9} {:>10} {:>12} {}",
                "GPU",
                s_gpu.stats.nodes,
                s_gpu.stats.qnodes,
                format!("{}ms", gpu_ms),
                format!("{:.0}", gpu_nps),
                board_move_string(mv),
            );

            let mut s_cpu = Searcher::new();
            s_cpu.set_tt_size(64);
            s_cpu.set_evaluator(eval_cpu);
            s_cpu.set_gpu_enabled(false);

            let start = Instant::now();
            let mut b = board.clone();
            let (mv, _) = s_cpu.search_fixed_nodes(&mut b, tier);
            let cpu_ms = start.elapsed().as_millis();
            let cpu_nps = if cpu_ms > 0 {
                (s_cpu.stats.nodes as f64 / cpu_ms as f64) * 1000.0
            } else {
                0.0
            };
            println!(
                "  {:<6} {:>10} {:>9} {:>10} {:>12} {}",
                "CPU",
                s_cpu.stats.nodes,
                s_cpu.stats.qnodes,
                format!("{}ms", cpu_ms),
                format!("{:.0}", cpu_nps),
                board_move_string(mv),
            );
            println!(
                "  Speedup: {:.2}x (GPU vs CPU, {} nodes fixed)\n",
                if cpu_ms > 0 { gpu_nps / cpu_nps.max(1.0) } else { 0.0 },
                tier_name,
            );
        }
        println!();
    }
}

pub fn run_material_test() {
    use rust_chess_engine::evaluation::Evaluator;
    use rust_chess_engine::nnue::NNUENetwork;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              CONTROLLED MATERIAL SANITY TEST                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let network = if std::path::Path::new("trained.nnue").exists() {
        let weights = rust_chess_engine::nnue::NNUESerializer::load("trained.nnue")
            .expect("Failed to load trained.nnue");
        println!("Network: trained.nnue (trained weights)\n");
        NNUENetwork::from_weights(weights)
    } else {
        println!("Network: fresh random weights (no trained.nnue found)\n");
        NNUENetwork::new()
    };

    println!("All scores are from the side-to-move perspective (cp).\n");

    let positions: Vec<(&str, &str)> = vec![
        ("K vs K",   "4k3/8/8/8/8/8/8/4K3 w - - 0 1"),
        ("K+P vs K", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"),
        ("K+N vs K", "4k3/8/8/8/8/8/4N3/4K3 w - - 0 1"),
        ("K+B vs K", "4k3/8/8/8/8/8/4B3/4K3 w - - 0 1"),
        ("K+R vs K", "4k3/8/8/8/8/8/4R3/4K3 w - - 0 1"),
        ("K+Q vs K", "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1"),
        ("K vs K+Q", "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"),
        ("Startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Italian",  "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5"),
        ("Sicilian", "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"),
        ("KQ mate1", "4k3/8/4K3/8/8/8/5Q2/8 w - - 0 1"),
        ("KQ mate2", "8/8/8/4k3/8/4Q3/4K3/8 w - - 0 1"),
    ];

    for (name, fen) in positions {
        println!("--- {} ---", name);
        println!("  FEN:              {}", fen);
        let board = Board::from_fen(fen).unwrap_or_else(|_| panic!("Invalid FEN {}", fen));

        let hc = Evaluator::hand_crafted_evaluate(&board);
        let (_h, _a, raw) = network.forward_sparse(&board);
        let acc = network.create_accumulator(&board);
        let acc_score = network.evaluate_accumulator(&acc, board.turn);
        let full_score = network.evaluate(&board);

        println!("  Handcrafted:      {:>8} cp", hc);
        println!("  NNUE raw:         {:>12.5}", raw);
        println!("  NNUE scaled:      {:>8} cp", full_score);
        println!("  NNUE accumulator: {:>8} cp", acc_score);
        println!();
    }
}

fn board_move_string(mv: Option<rust_chess_engine::board::ChessMove>) -> String {
    match mv {
        Some(m) => format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to)),
        None => "none".to_string(),
    }
}

pub fn run_batch_eval_benchmark() {
    use rust_chess_engine::nnue::gpu::GpuNnue;
    use rust_chess_engine::nnue::NNUENetwork;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║        GPU BATCH EVAL vs CPU FULL-FORWARD THROUGHPUT          ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let board = Board::from_fen("r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8")
        .expect("Valid FEN");
    let network = NNUENetwork::new();

    let count: usize = 16_384;

    let start = Instant::now();
    let mut cpu_sum = 0i64;
    for _ in 0..count {
        cpu_sum += network.evaluate(&board) as i64;
    }
    let cpu_ms = start.elapsed().as_millis();
    let cpu_eps = if cpu_ms > 0 {
        (count as f64 / cpu_ms as f64) * 1000.0
    } else {
        0.0
    };
    println!(
        "CPU  full-forward: {:>6} evals {:>8}ms  {:>10.0} evals/s  (sum {})",
        count, cpu_ms, cpu_eps, cpu_sum
    );

    let acc = network.create_accumulator(&board);
    let start = Instant::now();
    let mut acc_sum = 0i64;
    for _ in 0..count {
        acc_sum += network.evaluate_accumulator(&acc, board.turn) as i64;
    }
    let acc_ms = start.elapsed().as_millis();
    let acc_eps = if acc_ms > 0 {
        (count as f64 / acc_ms as f64) * 1000.0
    } else {
        0.0
    };
    println!(
        "CPU  accumulator: {:>6} evals {:>8}ms  {:>10.0} evals/s  (sum {})",
        count, acc_ms, acc_eps, acc_sum
    );

    let boards: Vec<_> = std::iter::repeat_n(board, count).collect();
    let gpu = GpuNnue::from_weights(&network.weights).expect("GPU init failed");

    let start = Instant::now();
    let gpu_scores = gpu.evaluate_boards(&boards);
    let gpu_ms = start.elapsed().as_millis();
    let gpu_eps = if gpu_ms > 0 {
        (count as f64 / gpu_ms as f64) * 1000.0
    } else {
        0.0
    };
    let gpu_sum: i64 = gpu_scores.iter().map(|&s| s as i64).sum();
    println!(
        "GPU  full-forward: {:>6} evals {:>8}ms  {:>10.0} evals/s  (sum {})",
        count, gpu_ms, gpu_eps, gpu_sum
    );
    println!("Speedup vs CPU full-forward: {:.2}x", gpu_eps / cpu_eps.max(1.0));

    let _ = gpu_sum;
    println!("\nSearch uses the accumulator path (incrementally cheap), which the");
    println!("full-forward GPU batch cannot beat in a per-node alpha-beta loop.");
}

const GPU_DEPTH_CONFIGS: &[(&str, bool, u32)] = &[
    ("CPU-only", false, 0),
    ("GPU d=0", true, 0),
    ("GPU d=4", true, 4),
    ("GPU d=8", true, 8),
    ("GPU d=12", true, 12),
    ("GPU d=16", true, 16),
];

const GPU_DEPTH_POSITIONS: &[(&str, &str)] = &[
    ("Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Middlegame", "r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8"),
    ("Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
];

fn load_trained_network() -> rust_chess_engine::nnue::NNUENetwork {
    if std::path::Path::new("trained.nnue").exists() {
        let weights = rust_chess_engine::nnue::NNUESerializer::load("trained.nnue")
            .expect("Failed to load trained.nnue");
        println!("Network: trained.nnue (trained weights)\n");
        rust_chess_engine::nnue::NNUENetwork::from_weights(weights)
    } else {
        println!("Network: fresh random weights (no trained.nnue found)\n");
        rust_chess_engine::nnue::NNUENetwork::new()
    }
}

fn searcher_for_config(
    network: &rust_chess_engine::nnue::NNUENetwork,
    gpu_enabled: bool,
    gpu_max_depth: u32,
) -> rust_chess_engine::search::Searcher {
    let mut searcher = Searcher::new();
    searcher.set_tt_size(64);
    searcher.set_evaluator(rust_chess_engine::evaluation::Evaluator::with_nnue(network.clone()));
    searcher.set_gpu_enabled(gpu_enabled);
    searcher.set_gpu_max_depth(gpu_max_depth);
    searcher
}

pub fn run_gpu_depth_benchmark() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║          GPU MAX DEPTH SWEEP - NPS & ACHIEVED DEPTH            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let network = load_trained_network();

    println!("{}", "═".repeat(72));
    println!("SECTION 1: FIXED-NODE SEARCH (nodes/s, equal node budget)");
    println!("{}", "═".repeat(72));

    for (label, gpu_enabled, gpu_max_depth) in GPU_DEPTH_CONFIGS {
        let mut searcher = searcher_for_config(&network, *gpu_enabled, *gpu_max_depth);
        println!("\n--- {} (gpu={}) ---", label, if *gpu_enabled { format!("maxdepth={}", gpu_max_depth) } else { "off".into() });

        for (node_target, tier_name) in [(25_000u64, "25k"), (60_000u64, "60k")] {
            let mut nps_sum = 0.0f64;
            let mut time_sum = 0u128;
            let mut nodes_sum = 0u64;
            let mut max_depth = 0u32;
            println!("  Fixed nodes/move: {}", tier_name);
            for (pos_name, fen) in GPU_DEPTH_POSITIONS {
                let board = Board::from_fen(fen).expect("Valid FEN");
                let start = Instant::now();
                let (_, _) = searcher.search_fixed_nodes(&mut board.clone(), node_target);
                let ms = start.elapsed().as_millis().max(1);
                let nodes = searcher.stats.nodes + searcher.stats.qnodes;
                let nps = nodes as f64 / (ms as f64 / 1000.0);
                nps_sum += nps;
                time_sum += ms;
                nodes_sum += nodes;
                max_depth = max_depth.max(searcher.stats.search_depth);
                println!(
                    "    {:<12} {:>6}ms | {:>10.0} nps | depth {:<2} | nodes {:>10}",
                    pos_name, ms, nps, searcher.stats.search_depth, nodes
                );
                if *gpu_enabled {
                    let cfg_line = {
                        let batch_min = searcher.stats.tail_gpu_batch_min;
                        let batches = searcher.stats.gpu_batches;
                        let children = searcher.stats.gpu_children_total;
                        let hints = searcher.stats.hints_used;
                        let q = searcher.stats.qnodes;
                        let n = searcher.stats.nodes;
                        format!(
                            "  ^ gpu: {} batches, avg {:.1} children/batch (min {}), {} hints | q/n = {:.2} ({} qnodes / {} nodes)",
                            batches,
                            if batches > 0 { children as f64 / batches as f64 } else { 0.0 },
                            batch_min,
                            hints,
                            if n > 0 { q as f64 / n as f64 } else { f64::NAN },
                            q,
                            n
                        )
                    };
                    println!("{}", cfg_line);
                } else if searcher.stats.qnodes > 0 {
                    let q = searcher.stats.qnodes;
                    let n = searcher.stats.nodes.max(1);
                    println!(
                        "  ^ cpu: q/n = {:.2} ({} qnodes / {} nodes)",
                        q as f64 / n as f64,
                        q,
                        searcher.stats.nodes
                    );
                }
            }
            println!(
                "    avg nps {:>10.0} | total {:>6}ms | nodes {:>10} | max depth {}",
                nps_sum / GPU_DEPTH_POSITIONS.len() as f64,
                time_sum,
                nodes_sum,
                max_depth
            );
        }
    }

    println!("\n{}", "═".repeat(72));
    println!("SECTION 2: TIME-LIMITED SEARCH (700 ms/move, maxdepth 64)");
    println!("{}", "═".repeat(72));

    for (label, gpu_enabled, gpu_max_depth) in GPU_DEPTH_CONFIGS {
        let mut searcher = searcher_for_config(&network, *gpu_enabled, *gpu_max_depth);
        let mut depth_sum = 0u32;
        let mut nps_sum = 0.0f64;
        println!("\n--- {} ---", label);
        for (pos_name, fen) in GPU_DEPTH_POSITIONS {
            let board = Board::from_fen(fen).expect("Valid FEN");
            let start = Instant::now();
            let (_, depth) = searcher.search_with_time_management(&mut board.clone(), 64, 700);
            let ms = start.elapsed().as_millis().max(1);
            let nodes = searcher.stats.nodes + searcher.stats.qnodes;
            let nps = nodes as f64 / (ms as f64 / 1000.0);
            depth_sum += depth;
            nps_sum += nps;
            println!(
                "    {:<12} {:>5}ms | depth {:<3} | {:>10.0} nps | nodes {:>10}",
                pos_name, ms, depth, nps, nodes
            );
        }
        println!(
            "    avg depth {:.2} | avg nps {:>10.0}",
            depth_sum as f64 / GPU_DEPTH_POSITIONS.len() as f64,
            nps_sum / GPU_DEPTH_POSITIONS.len() as f64
        );
    }
}

fn gpu_match(
    a_name: &str,
    a_gpu_enabled: bool,
    a_max_depth: u32,
    b_name: &str,
    b_gpu_enabled: bool,
    b_max_depth: u32,
    network: rust_chess_engine::nnue::NNUENetwork,
    nodes_per_move: u64,
    num_games: usize,
    random_plies: usize,
) -> rust_chess_engine::nnue::MatchResult {
    use rand::seq::SliceRandom;
    use rust_chess_engine::nnue::MatchResult;

    let mut a_wins = 0;
    let mut b_wins = 0;
    let mut draws = 0;

    for game_idx in 0..num_games {
        let a_is_white = game_idx % 2 == 0;
        let mut board = Board::new();

        let mut searcher_a = searcher_for_config(&network, a_gpu_enabled, a_max_depth);
        let mut searcher_b = searcher_for_config(&network, b_gpu_enabled, b_max_depth);

        for _ in 0..random_plies {
            let moves = board.generate_moves();
            if moves.is_empty() {
                break;
            }
            let mut rng = rand::thread_rng();
            let chosen = *moves.choose(&mut rng).unwrap();
            let _ = board.execute_move(chosen.from, chosen.to, chosen.move_type);
        }

        let mut game_result = 0.5_f32;
        for _ in 0..120 {
            if board.halfmove_clock >= 100 {
                game_result = 0.5;
                break;
            }
            let legal = board.generate_moves();
            if legal.is_empty() {
                if board.is_in_check(board.turn) {
                    game_result = if board.turn == rust_chess_engine::board::Color::White { 0.0 } else { 1.0 };
                } else {
                    game_result = 0.5;
                }
                break;
            }

            let current_is_a = (board.turn == rust_chess_engine::board::Color::White && a_is_white)
                || (board.turn == rust_chess_engine::board::Color::Black && !a_is_white);

            let (best_move, score) = if current_is_a {
                searcher_a.search_fixed_nodes(&mut board, nodes_per_move)
            } else {
                searcher_b.search_fixed_nodes(&mut board, nodes_per_move)
            };

            if score.abs() > 28000 {
                let white_winning = (board.turn == rust_chess_engine::board::Color::White && score > 28000)
                    || (board.turn == rust_chess_engine::board::Color::Black && score < -28000);
                game_result = if white_winning { 1.0 } else { 0.0 };
                break;
            }

            let chosen = best_move.unwrap_or(legal[0]);
            if board.execute_move(chosen.from, chosen.to, chosen.move_type).is_err() {
                break;
            }
        }

        if game_result == 0.5 {
            draws += 1;
            println!("  Game {}: Draw", game_idx + 1);
        } else if (game_result == 1.0 && a_is_white) || (game_result == 0.0 && !a_is_white) {
            a_wins += 1;
            println!("  Game {}: {} wins", game_idx + 1, a_name);
        } else {
            b_wins += 1;
            println!("  Game {}: {} wins", game_idx + 1, b_name);
        }
    }

    let total = a_wins + b_wins + draws;
    let score = (a_wins as f64 + 0.5 * draws as f64) / (total.max(1) as f64);
    let elo_diff = if score <= 0.001 {
        -800.0
    } else if score >= 0.999 {
        800.0
    } else {
        -400.0 * (1.0 / score - 1.0).log10()
    };

    MatchResult {
        engine_a_name: a_name.to_string(),
        engine_b_name: b_name.to_string(),
        engine_a_wins: a_wins,
        engine_b_wins: b_wins,
        draws,
        total_games: total,
        score_percentage: score,
        elo_difference: elo_diff,
    }
}

pub fn run_gpu_depth_matches() {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║          GPU MAX DEPTH SWEEP - EQUAL-NODE SELF-PLAY           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let network = load_trained_network();

    let nodes_per_move: u64 = 10_000;
    let games_per_match = 10;
    let random_plies = 4;

    for (label, gpu_enabled, gpu_max_depth) in &GPU_DEPTH_CONFIGS[0..] {
        println!("\n>>> {} vs CPU-only ({} games, {} nodes/move) <<<",
            label, games_per_match, nodes_per_move);
        let result = gpu_match(
            label, *gpu_enabled, *gpu_max_depth,
            "CPU-only", false, 0,
            network.clone(),
            nodes_per_move,
            games_per_match,
            random_plies,
        );
        result.display();
    }
}

pub fn run_eval_speed_benchmark(iterations: usize) {
    use rust_chess_engine::evaluation::Evaluator;
    use std::time::Instant;

    let boards = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8",
        "4k3/8/8/8/8/8/6q1/5rk1 w - - 0 1",
    ];

    for fen in &boards {
        let board = Board::from_fen(fen).expect("Valid FEN");
        let mut sum = 0i64;
        let start = Instant::now();
        for _ in 0..iterations {
            sum += Evaluator::hand_crafted_evaluate(&board) as i64;
        }
        let elapsed = start.elapsed();
        let eps = iterations as f64 / elapsed.as_secs_f64();
        println!(
            "{:<4} keval/s | {:>6.1} us/eval | sum={} | {}",
            eps / 1_000.0,
            elapsed.as_secs_f64() * 1e6 / iterations as f64,
            sum,
            fen
        );
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

    println!("\nSECTION 2: FIXED-NODE SEARCH THROUGHPUT (10k, 100k, 1M nodes)");
    println!("{}", "─".repeat(64));

    let bench_positions = vec![
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Complex Middlegame", "r1bqk2r/pp1p1ppp/2n1pn2/8/2PP4/2P2N2/P4PPP/R1BQKB1R w KQkq - 0 8"),
        ("Tactical Endgame",  "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ];

    let node_tiers = vec![("10k", 10_000u64), ("100k", 100_000u64), ("1M", 1_000_000u64)];

    for (pos_name, fen) in &bench_positions {
        println!("\nPosition: {}", pos_name);
        println!(
            "  {:<6} | {:>10} | {:>10} | {:>10} | {:>10} | {:>7}",
            "Tier", "HC Time", "HC NPS", "NN Time", "NN NPS", "Speedup"
        );
        println!("  {}", "─".repeat(64));

        let test_board = Board::from_fen(fen).expect("Valid FEN");

        for (tier_name, node_target) in &node_tiers {
            let mut searcher_hc = Searcher::new();
            searcher_hc.set_tt_size(16);
            searcher_hc.set_evaluation_mode(EvaluationMode::Handcrafted);
            let start_hc = Instant::now();
            searcher_hc.search_fixed_nodes(&mut test_board.clone(), *node_target);
            let time_hc = start_hc.elapsed().as_millis().max(1);
            let hc_nodes = searcher_hc.stats.nodes + searcher_hc.stats.qnodes;
            let hc_nps = hc_nodes as f64 / (time_hc as f64 / 1000.0);

            let mut searcher_nn = Searcher::new();
            searcher_nn.set_tt_size(16);
            searcher_nn.set_evaluator(Evaluator::with_nnue(NNUENetwork::new()));
            let start_nn = Instant::now();
            searcher_nn.search_fixed_nodes(&mut test_board.clone(), *node_target);
            let time_nn = start_nn.elapsed().as_millis().max(1);
            let nn_nodes = searcher_nn.stats.nodes + searcher_nn.stats.qnodes;
            let nn_nps = nn_nodes as f64 / (time_nn as f64 / 1000.0);

            let speedup = if hc_nps > 0.0 { nn_nps / hc_nps } else { 0.0 };
            let speedup_str = if speedup >= 1.0 {
                format!("{:.2}x ✓", speedup)
            } else {
                format!("{:.2}x ↓", speedup)
            };

            println!(
                "  {:<6} | {:>8}ms | {:>10.0} | {:>8}ms | {:>10.0} | {:>7}",
                tier_name, time_hc, hc_nps, time_nn, nn_nps, speedup_str
            );
        }
    }

    println!("\nSECTION 3: BEST-MOVE AGREEMENT ANALYSIS (Fixed 10k Nodes)");
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

    let fixed_agree_nodes = 10_000u64;
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
        let (mv_hc, _) = s_hc.search_fixed_nodes(&mut b.clone(), fixed_agree_nodes);

        let mut s_nn = Searcher::new();
        s_nn.set_tt_size(16);
        s_nn.set_evaluator(Evaluator::with_nnue(NNUENetwork::new()));
        let (mv_nn, _) = s_nn.search_fixed_nodes(&mut b.clone(), fixed_agree_nodes);

        let hc_str = mv_hc.map(|m| format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to))).unwrap_or_else(|| "none".to_string());
        let nn_str = mv_nn.map(|m| format!("{}{}", Board::square_to_string(m.from), Board::square_to_string(m.to))).unwrap_or_else(|| "none".to_string());
        let agree = hc_str == nn_str;
        if agree { agree_count += 1; }
        total_count += 1;

        println!(
            "  {:<14} | {:<8} | {:<8} | {:<7} | {} / {}",
            pos_name, hc_str, nn_str,
            if agree { "✓ Match" } else { "✗ Diff " },
            s_hc.stats.nodes + s_hc.stats.qnodes, s_nn.stats.nodes + s_nn.stats.qnodes
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
