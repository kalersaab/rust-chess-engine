use crate::board::Board;
use crate::board::chess_move::ChessMove;
use crate::evaluation::{Evaluator, Score};
use crate::nnue::NNUEAccumulator;
use super::alphabeta::AlphaBeta;
use super::aspiration::AspirationWindows;
use crate::transposition_table::TranspositionTable;
use crate::move_ordering::MoveOrderer;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn search_one_depth(
    ab: &mut AlphaBeta,
    aspiration: &mut AspirationWindows,
    board: &mut Board,
    depth: u32,
    prev_score: Score,
    tt: &TranspositionTable,
    orderer: &mut MoveOrderer,
    evaluator: &Evaluator,
    accumulator: Option<&NNUEAccumulator>,
) -> Score {
    ab.best_move_at_root = None;
    ab.root_depth = 0;
    aspiration.search_with_eval(ab, board, depth, prev_score, tt, orderer, evaluator, accumulator)
}

pub struct IterativeDeepening {
    pub best_move: Option<ChessMove>,
    pub best_score: Score,
    pub depth_achieved: u32,
    pub time_spent_ms: u128,
    pub ab: AlphaBeta,
    pub stop_flag: Option<Arc<AtomicBool>>,
    pub gpu_enabled: bool,
    pub gpu_batch_min: usize,
    pub gpu_max_depth: u32,
    pub gpu: Option<Arc<crate::nnue::gpu::GpuNnue>>,
    gpu_checked: bool,
}

impl IterativeDeepening {
    pub fn new() -> Self {
        IterativeDeepening {
            best_move: None,
            best_score: 0,
            depth_achieved: 0,
            time_spent_ms: 0,
            ab: AlphaBeta::new(),
            stop_flag: None,
            gpu_enabled: false,
            gpu_batch_min: 8,
            gpu_max_depth: 16,
            gpu: None,
            gpu_checked: false,
        }
    }

    pub fn set_gpu_enabled(&mut self, enabled: bool) {
        self.gpu_enabled = enabled;
        if !enabled {
            self.gpu = None;
            self.gpu_checked = false;
        }
    }

    pub fn set_gpu_batch_min(&mut self, min: usize) {
        self.gpu_batch_min = min.max(2);
    }

    pub fn set_gpu_max_depth(&mut self, depth: u32) {
        self.gpu_max_depth = depth;
    }

    pub fn invalidate_gpu(&mut self) {
        self.gpu = None;
        self.gpu_checked = false;
    }

    fn ensure_gpu(&mut self, evaluator: &crate::evaluation::Evaluator) {
        if !self.gpu_enabled || self.gpu_checked {
            return;
        }
        self.gpu_checked = true;
        if matches!(evaluator.mode, crate::evaluation::EvaluationMode::NNUE) {
            if let Some(net) = &evaluator.nnue_network {
                self.gpu = crate::nnue::gpu::GpuNnue::from_weights(&net.weights)
                    .map(std::sync::Arc::new);
            }
        }
    }

    pub fn set_stop_flag(&mut self, flag: Arc<AtomicBool>) {
        self.stop_flag = Some(Arc::clone(&flag));
    }

    fn stop_requested(&self) -> bool {
        if let Some(ref flag) = self.stop_flag {
            return flag.load(Ordering::Relaxed);
        }
        false
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_limit_ms: Option<u128>,
    ) -> Option<ChessMove> {
        let evaluator = crate::evaluation::Evaluator::new();
        self.search_with_eval(board, max_depth, time_limit_ms, &evaluator)
    }

    pub fn search_with_eval(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_limit_ms: Option<u128>,
        evaluator: &crate::evaluation::Evaluator,
    ) -> Option<ChessMove> {
        let start = Instant::now();
        let time_limit = time_limit_ms.map(|ms| Duration::from_millis(ms as u64));

        let moves = board.generate_moves();
        if moves.is_empty() {
            return None;
        }

        let tt = TranspositionTable::new(16);
        let mut orderer = MoveOrderer::new(max_depth);
        self.ensure_gpu(evaluator);
        self.ab = AlphaBeta::new();
        self.ab.gpu = self.gpu.clone();
        self.ab.gpu_batch_min = self.gpu_batch_min;
        self.ab.gpu_max_depth = self.gpu_max_depth;
        if let Some(ref flag) = self.stop_flag {
            self.ab.set_stop_flag(Arc::clone(flag));
        }
        let mut aspiration = AspirationWindows::new();

        let root_accumulator = if evaluator.mode != crate::evaluation::EvaluationMode::Handcrafted {
            evaluator.nnue_network.as_ref().map(|net| net.create_accumulator(board))
        } else {
            None
        };

        self.best_move = Some(moves[0]);

        for depth in 1..=max_depth {
            if self.stop_requested() {
                break;
            }
            if let Some(limit) = time_limit {
                if start.elapsed() > limit {
                    break;
                }
            }

            let score = search_one_depth(
                &mut self.ab,
                &mut aspiration,
                board,
                depth,
                self.best_score,
                &tt,
                &mut orderer,
                evaluator,
                root_accumulator.as_ref(),
            );
            
            if self.ab.is_aborted || self.stop_requested() {
                break;
            }

            self.best_score = score;
            self.depth_achieved = depth;

            if let Some(best) = self.ab.best_move_at_root {
                self.best_move = Some(best);
            }

            if let Some(limit) = time_limit {
                if start.elapsed() > limit {
                    break;
                }
            }
        }

        self.time_spent_ms = start.elapsed().as_millis();
        self.best_move
    }

    pub fn search_parallel_with_eval(
        &mut self,
        board: &mut Board,
        max_depth: u32,
        time_limit_ms: Option<u128>,
        threads: usize,
        evaluator: &Evaluator,
    ) -> Option<ChessMove> {
        let start = Instant::now();
        let time_limit = time_limit_ms.map(|ms| Duration::from_millis(ms as u64));
        let thread_count = threads.clamp(1, 16);

        let moves = board.generate_moves();
        if moves.is_empty() {
            return None;
        }

        self.ensure_gpu(evaluator);
        let gpu_batch_min = self.gpu_batch_min;
        let gpu_max_depth = self.gpu_max_depth;
        let outer_stop = self.stop_flag.clone();

        let shared_tt = Arc::new(TranspositionTable::new(16));
        let shared_best_depth = Arc::new(AtomicU32::new(0));
        let shared_best_move = Arc::new(Mutex::new(Some(moves[0])));
        let shared_best_score = Arc::new(Mutex::new(0));
        let sum_nodes = Arc::new(AtomicU64::new(0));
        let sum_qnodes = Arc::new(AtomicU64::new(0));
        let sum_cutoffs = Arc::new(AtomicU64::new(0));

        let root_accumulator = if evaluator.mode != crate::evaluation::EvaluationMode::Handcrafted {
            evaluator.nnue_network.as_ref().map(|net| net.create_accumulator(board))
        } else {
            None
        };

        let first_move = moves[0];

        std::thread::scope(|scope| {
            for _ in 0..thread_count {
                let mut local_board = board.clone();
                let mut local_ab = AlphaBeta::new();
                local_ab.gpu_batch_min = gpu_batch_min;
                local_ab.gpu_max_depth = gpu_max_depth;
                let local_stop = outer_stop.clone();
                if let Some(ref flag) = local_stop {
                    local_ab.set_stop_flag(Arc::clone(flag));
                }
                let mut local_orderer = MoveOrderer::new(max_depth);
                let mut local_aspiration = AspirationWindows::new();
                let local_acc = root_accumulator.clone();

                let tt = Arc::clone(&shared_tt);
                let best_depth = Arc::clone(&shared_best_depth);
                let best_move = Arc::clone(&shared_best_move);
                let best_score = Arc::clone(&shared_best_score);
                let acc_nodes = Arc::clone(&sum_nodes);
                let acc_qnodes = Arc::clone(&sum_qnodes);
                let acc_cutoffs = Arc::clone(&sum_cutoffs);

                scope.spawn(move || {
                    let mut my_best = Some(first_move);
                    let mut prev_score = 0;

                    for depth in 1..=max_depth {
                        if let Some(ref f) = local_stop {
                            if f.load(Ordering::Relaxed) {
                                break;
                            }
                        }
                        if let Some(limit) = time_limit {
                            if start.elapsed() > limit {
                                break;
                            }
                        }

                        let score = search_one_depth(
                            &mut local_ab,
                            &mut local_aspiration,
                            &mut local_board,
                            depth,
                            prev_score,
                            &tt,
                            &mut local_orderer,
                            evaluator,
                            local_acc.as_ref(),
                        );

                        if local_ab.is_aborted {
                            break;
                        }
                        if let Some(ref f) = local_stop {
                            if f.load(Ordering::Relaxed) {
                                break;
                            }
                        }

                        prev_score = score;
                        if let Some(best) = local_ab.best_move_at_root {
                            my_best = Some(best);
                        }

                        let mut current = best_depth.load(Ordering::Acquire);
                        loop {
                            if depth <= current {
                                break;
                            }
                            match best_depth.compare_exchange_weak(
                                current,
                                depth,
                                Ordering::AcqRel,
                                Ordering::Acquire,
                            ) {
                                Ok(_) => {
                                    *best_move.lock().expect("best move lock") = my_best;
                                    *best_score.lock().expect("best score lock") = score;
                                    break;
                                }
                                Err(actual) => current = actual,
                            }
                        }
                    }

                    acc_nodes.fetch_add(local_ab.nodes, Ordering::Relaxed);
                    acc_qnodes.fetch_add(local_ab.qnodes, Ordering::Relaxed);
                    acc_cutoffs.fetch_add(local_ab.cutoffs, Ordering::Relaxed);
                });
            }
        });

        self.best_move = *shared_best_move.lock().expect("best move lock");
        self.best_score = *shared_best_score.lock().expect("best score lock");
        self.depth_achieved = shared_best_depth.load(Ordering::Acquire);
        self.time_spent_ms = start.elapsed().as_millis();
        self.ab.nodes = sum_nodes.load(Ordering::Relaxed);
        self.ab.qnodes = sum_qnodes.load(Ordering::Relaxed);
        self.ab.cutoffs = sum_cutoffs.load(Ordering::Relaxed);

        self.best_move
    }

    pub fn search_with_node_limit(
        &mut self,
        board: &mut Board,
        max_nodes: u64,
        evaluator: &crate::evaluation::Evaluator,
    ) -> Option<ChessMove> {
        let start = Instant::now();
        let moves = board.generate_moves();
        if moves.is_empty() {
            return None;
        }

        let tt = TranspositionTable::new(16);
        let mut orderer = MoveOrderer::new(64);
        self.ensure_gpu(evaluator);
        self.ab = AlphaBeta::new();
        self.ab.gpu = self.gpu.clone();
        self.ab.gpu_batch_min = self.gpu_batch_min;
        self.ab.gpu_max_depth = self.gpu_max_depth;
        if let Some(ref flag) = self.stop_flag {
            self.ab.set_stop_flag(Arc::clone(flag));
        }
        self.ab.node_limit = Some(max_nodes);
        let mut aspiration = AspirationWindows::new();

        let root_accumulator = if evaluator.mode != crate::evaluation::EvaluationMode::Handcrafted {
            evaluator.nnue_network.as_ref().map(|net| net.create_accumulator(board))
        } else {
            None
        };

        self.best_move = Some(moves[0]);

        for depth in 1..=64 {
            if self.ab.total_nodes() >= max_nodes || self.ab.is_aborted || self.stop_requested() {
                break;
            }

            self.ab.best_move_at_root = None;
            self.ab.root_depth = 0;

            let score = aspiration.search_with_eval(
                &mut self.ab,
                board,
                depth,
                self.best_score,
                &tt,
                &mut orderer,
                evaluator,
                root_accumulator.as_ref(),
            );

            if !self.ab.is_aborted {
                self.best_score = score;
                self.depth_achieved = depth;
                if let Some(best) = self.ab.best_move_at_root {
                    self.best_move = Some(best);
                }
            } else {
                if let Some(best) = self.ab.best_move_at_root {
                    self.best_move = Some(best);
                }
                break;
            }
        }

        self.time_spent_ms = start.elapsed().as_millis();
        self.best_move
    }
}
