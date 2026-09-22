use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use crate::board::{Board, Color};
use crate::evaluation::{EvaluationMode, Evaluator, Score};
use crate::nnue::network::NNUENetwork;
use crate::nnue::pgn_loader::GamePosition;
use crate::nnue::trainer::sigmoid;
use crate::search::Searcher;

#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub nodes_per_move: u64,
    pub random_opening_plies: usize,
    pub max_moves: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        SelfPlayConfig {
            nodes_per_move: 10_000,
            random_opening_plies: 6,
            max_moves: 120,
        }
    }
}

pub struct SelfPlayGenerator {
    pub config: SelfPlayConfig,
}

impl SelfPlayGenerator {
    pub fn new(config: SelfPlayConfig) -> Self {
        SelfPlayGenerator { config }
    }

    /// Generate self-play games using classical search with random opening moves,
    /// returning labeled positions (FEN and win probability target).
    pub fn generate_games(&self, num_games: usize) -> Vec<GamePosition> {
        let mut all_positions = Vec::new();

        for game_idx in 0..num_games {
            let mut board = Board::new();
            let mut searcher = Searcher::new();
            searcher.set_evaluation_mode(EvaluationMode::Handcrafted);

            let mut game_history: Vec<(String, Color, Score)> = Vec::new();
            let mut result = 0.5_f32; // Default draw

            for ply in 0..self.config.max_moves {
                let legal_moves = board.generate_moves();
                if legal_moves.is_empty() {
                    if board.is_in_check(board.turn) {
                        // Checkmate: side to move lost
                        result = if board.turn == Color::White { 0.0 } else { 1.0 };
                    } else {
                        // Stalemate
                        result = 0.5;
                    }
                    break;
                }

                if board.halfmove_clock >= 100 {
                    result = 0.5;
                    break;
                }

                let fen = board.to_fen();

                let chosen_move = if ply < self.config.random_opening_plies {
                    // Random opening ply to induce diversity
                    use rand::seq::SliceRandom;
                    let mut rng = rand::thread_rng();
                    *legal_moves.choose(&mut rng).unwrap()
                } else {
                    let (best_move, score) = searcher.search_fixed_nodes(&mut board, self.config.nodes_per_move);
                    game_history.push((fen, board.turn, score));

                    // Early adjudication if one side is overwhelmingly ahead
                    if score.abs() > 2500 {
                        let white_winning = (board.turn == Color::White && score > 2500)
                            || (board.turn == Color::Black && score < -2500);
                        result = if white_winning { 1.0 } else { 0.0 };
                        break;
                    }

                    best_move.unwrap_or(legal_moves[0])
                };

                if board.execute_move(chosen_move.from, chosen_move.to, chosen_move.move_type).is_err() {
                    break;
                }
            }

            // Convert game positions to training samples
            // Label is a 50/50 blend of final game outcome and search evaluation target
            for (fen, turn, score) in game_history {
                let white_eval = if turn == Color::White { score } else { -score };
                let search_prob = sigmoid(white_eval as f32 / 400.0);
                let blended_target = 0.5 * result + 0.5 * search_prob;

                all_positions.push(GamePosition {
                    fen,
                    result: blended_target,
                });
            }

            println!(
                "  Game {}/{}: finished with result {:.1} ({} positions)",
                game_idx + 1,
                num_games,
                result,
                all_positions.len()
            );
        }

        all_positions
    }

    /// Label an existing EPD dataset by evaluating positions with fixed-node classical search.
    pub fn label_epd_file<P: AsRef<Path>>(
        &self,
        epd_path: P,
        max_positions: usize,
    ) -> Result<Vec<GamePosition>, String> {
        let file = File::open(epd_path.as_ref())
            .map_err(|e| format!("Failed to open EPD file: {}", e))?;
        let reader = BufReader::new(file);

        let mut searcher = Searcher::new();
        searcher.set_evaluation_mode(EvaluationMode::Handcrafted);

        let mut positions = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if positions.len() >= max_positions {
                break;
            }

            let line = line.map_err(|e| format!("Error reading line: {}", e))?;
            let fen = line.trim();
            if fen.is_empty() {
                continue;
            }

            if let Ok(mut board) = Board::from_fen(fen) {
                let (_best_move, score) = searcher.search_fixed_nodes(&mut board, self.config.nodes_per_move);
                let white_eval = if board.turn == Color::White { score } else { -score };
                let target_prob = sigmoid(white_eval as f32 / 400.0);

                positions.push(GamePosition {
                    fen: fen.to_string(),
                    result: target_prob,
                });

                if (i + 1) % 100 == 0 || positions.len() == max_positions {
                    println!("  Scored {}/{} EPD positions", positions.len(), max_positions);
                }
            }
        }

        Ok(positions)
    }
}

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub engine_a_name: String,
    pub engine_b_name: String,
    pub engine_a_wins: usize,
    pub engine_b_wins: usize,
    pub draws: usize,
    pub total_games: usize,
    pub score_percentage: f64,
    pub elo_difference: f64,
}

impl MatchResult {
    pub fn display(&self) {
        println!("\n{}", "=".repeat(60));
        println!("MATCH RESULT: {} vs {}", self.engine_a_name, self.engine_b_name);
        println!("{}", "=".repeat(60));
        println!("Total Games: {}", self.total_games);
        println!("{}: {} wins", self.engine_a_name, self.engine_a_wins);
        println!("{}: {} wins", self.engine_b_name, self.engine_b_wins);
        println!("Draws: {}", self.draws);
        println!("Score: {:.1}% for {}", self.score_percentage * 100.0, self.engine_a_name);
        println!("Estimated ΔElo: {:+.1}", self.elo_difference);
        println!("{}\n", "=".repeat(60));
    }
}

pub struct MatchRunner {
    pub nodes_per_move: u64,
    pub max_moves: usize,
}

impl MatchRunner {
    pub fn new(nodes_per_move: u64) -> Self {
        MatchRunner {
            nodes_per_move,
            max_moves: 120,
        }
    }

    /// Play an N-game match between Engine A and Engine B at fixed nodes per move.
    /// Alternates colors each game.
    pub fn run_match(
        &self,
        num_games: usize,
        engine_a_name: &str,
        engine_a_mode: EvaluationMode,
        engine_a_network: Option<NNUENetwork>,
        engine_b_name: &str,
        engine_b_mode: EvaluationMode,
        engine_b_network: Option<NNUENetwork>,
    ) -> MatchResult {
        let mut a_wins = 0;
        let mut b_wins = 0;
        let mut draws = 0;

        for game_idx in 0..num_games {
            // Alternate colors: even games A is White, odd games B is White
            let a_is_white = game_idx % 2 == 0;

            let mut board = Board::new();

            // Set up searchers with respective evaluators
            let mut searcher_a = Searcher::new();
            searcher_a.set_evaluation_mode(engine_a_mode);
            if let Some(ref net) = engine_a_network {
                let mut eval = Evaluator::new();
                eval.set_mode(engine_a_mode);
                eval.set_nnue_network(net.clone());
                searcher_a.set_evaluator(eval);
            }

            let mut searcher_b = Searcher::new();
            searcher_b.set_evaluation_mode(engine_b_mode);
            if let Some(ref net) = engine_b_network {
                let mut eval = Evaluator::new();
                eval.set_mode(engine_b_mode);
                eval.set_nnue_network(net.clone());
                searcher_b.set_evaluator(eval);
            }

            // Play random 4 plies to open different lines
            for _ in 0..4 {
                let moves = board.generate_moves();
                if moves.is_empty() {
                    break;
                }
                use rand::seq::SliceRandom;
                let mut rng = rand::thread_rng();
                let chosen = *moves.choose(&mut rng).unwrap();
                let _ = board.execute_move(chosen.from, chosen.to, chosen.move_type);
            }

            let mut game_result = 0.5_f32;

            for _move_num in 0..self.max_moves {
                let legal_moves = board.generate_moves();
                if legal_moves.is_empty() {
                    if board.is_in_check(board.turn) {
                        game_result = if board.turn == Color::White { 0.0 } else { 1.0 };
                    } else {
                        game_result = 0.5;
                    }
                    break;
                }

                if board.halfmove_clock >= 100 {
                    game_result = 0.5;
                    break;
                }

                let current_is_a = (board.turn == Color::White && a_is_white)
                    || (board.turn == Color::Black && !a_is_white);

                let (best_move, score) = if current_is_a {
                    searcher_a.search_fixed_nodes(&mut board, self.nodes_per_move)
                } else {
                    searcher_b.search_fixed_nodes(&mut board, self.nodes_per_move)
                };

                // Adjudicate mate or massive score
                if score.abs() > 28000 {
                    let white_winning = (board.turn == Color::White && score > 28000)
                        || (board.turn == Color::Black && score < -28000);
                    game_result = if white_winning { 1.0 } else { 0.0 };
                    break;
                }

                let chosen = best_move.unwrap_or(legal_moves[0]);
                if board.execute_move(chosen.from, chosen.to, chosen.move_type).is_err() {
                    break;
                }
            }

            // Determine game winner
            if game_result == 0.5 {
                draws += 1;
                println!("  Game {}: Draw (1/2 - 1/2)", game_idx + 1);
            } else if (game_result == 1.0 && a_is_white) || (game_result == 0.0 && !a_is_white) {
                a_wins += 1;
                println!("  Game {}: {} won as {}", game_idx + 1, engine_a_name, if a_is_white { "White" } else { "Black" });
            } else {
                b_wins += 1;
                println!("  Game {}: {} won as {}", game_idx + 1, engine_b_name, if !a_is_white { "White" } else { "Black" });
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
            engine_a_name: engine_a_name.to_string(),
            engine_b_name: engine_b_name.to_string(),
            engine_a_wins: a_wins,
            engine_b_wins: b_wins,
            draws,
            total_games: total,
            score_percentage: score,
            elo_difference: elo_diff,
        }
    }
}
