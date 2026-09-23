use crate::board::{Board, ChessMove};
use crate::evaluation::{EvaluationMode, Evaluator, Score};
use crate::nnue::gpu::GpuNnue;
use crate::nnue::NNUEAccumulator;
use crate::transposition_table::{TranspositionTable, BoundType};
use crate::move_ordering::MoveOrderer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct AlphaBeta {
    pub nodes: u64,
    pub qnodes: u64,
    pub cutoffs: u64,
    pub best_move_at_root: Option<ChessMove>,
    pub root_depth: u32,
    pub node_limit: Option<u64>,
    pub is_aborted: bool,
    pub stop_flag: Option<Arc<AtomicBool>>,
    pub gpu: Option<Arc<GpuNnue>>,
    pub gpu_batch_min: usize,
    pub gpu_max_depth: u32,
    pub gpu_batches: u64,
    pub gpu_children_total: u64,
    pub hints_used: u64,
}

impl AlphaBeta {
    pub fn new() -> Self {
        AlphaBeta {
            nodes: 0,
            qnodes: 0,
            cutoffs: 0,
            best_move_at_root: None,
            root_depth: 0,
            node_limit: None,
            is_aborted: false,
            stop_flag: None,
            gpu: None,
            gpu_batch_min: 8,
            gpu_max_depth: 16,
            gpu_batches: 0,
            gpu_children_total: 0,
            hints_used: 0,
        }
    }

    pub fn set_stop_flag(&mut self, flag: Arc<AtomicBool>) {
        self.stop_flag = Some(flag);
    }

    fn must_abort(&mut self) -> bool {
        if self.is_aborted {
            return true;
        }
        if let Some(ref flag) = self.stop_flag {
            if flag.load(Ordering::Relaxed) {
                self.is_aborted = true;
                return true;
            }
        }
        false
    }

    pub fn total_nodes(&self) -> u64 {
        self.nodes + self.qnodes
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        depth: u32,
        alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
    ) -> Score {
        let evaluator = Evaluator::new();
        self.search_with_eval(board, depth, alpha, beta, tt, orderer, &evaluator, None, None)
    }

    fn gpu_useable(
        &self,
        evaluator: &Evaluator,
        n_moves: usize,
        depth: u32,
    ) -> bool {
        self.gpu.is_some()
            && matches!(evaluator.mode, EvaluationMode::NNUE)
            && evaluator.nnue_network.is_some()
            && n_moves >= self.gpu_batch_min
            && depth <= self.gpu_max_depth
    }

    fn gpu_batch_children(
        &self,
        board: &Board,
        moves: &[ChessMove],
    ) -> Vec<(Option<Board>, Option<Score>)> {
        debug_assert!(self.gpu.is_some());
        let gpu = self.gpu.as_ref().expect("gpu checked");

        let mut children = Vec::with_capacity(moves.len());
        let mut eval_boards = Vec::with_capacity(moves.len());
        for mv in moves {
            let mut b = board.clone();
            if b.execute_move(mv.from, mv.to, mv.move_type).is_ok() {
                eval_boards.push(b.clone());
                children.push(Some(b));
            } else {
                children.push(None);
            }
        }

        let scores = gpu.evaluate_boards(&eval_boards);
        let mut aligned = Vec::with_capacity(moves.len());
        let mut si = 0;
        for c in children {
            if c.is_some() {
                aligned.push((c, scores.get(si).copied()));
                si += 1;
            } else {
                aligned.push((c, None));
            }
        }
        aligned
    }

    pub fn search_with_eval(
        &mut self,
        board: &mut Board,
        depth: u32,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
        evaluator: &Evaluator,
        accumulator: Option<&NNUEAccumulator>,
        stand_pat_hint: Option<Score>,
    ) -> Score {
        if self.must_abort() {
            return 0;
        }

        if let Some(limit) = self.node_limit {
            if self.total_nodes() >= limit {
                self.is_aborted = true;
                return 0;
            }
        }

        if depth > self.root_depth {
            self.root_depth = depth;
        }
        let is_root = depth == self.root_depth;
        if depth == 0 {
            return self.quiescence(board, alpha, beta, tt, orderer, evaluator, accumulator, stand_pat_hint);
        }

        self.nodes += 1;

        let hash_move = tt.lookup(board, depth).map(|_entry| {
            ChessMove::new(0, 0, 0, 0)
        });

        let mut moves = board.generate_moves();
        
        if moves.is_empty() {
            if board.is_in_check(board.turn) {
                return -100000 + (20 - depth) as Score;
            } else {
                return 0;
            }
        }

        orderer.sort_moves(&mut moves, board, depth, hash_move);

        let mut gpu_children: Option<Vec<(Option<Board>, Option<Score>)>> = None;
        if self.gpu_useable(evaluator, moves.len(), depth) {
            self.gpu_batches += 1;
            self.gpu_children_total += moves.len() as u64;
            let mut children = self.gpu_batch_children(board, &moves);
            let mut order: Vec<usize> = (0..moves.len()).collect();
            order.sort_by(|&a, &b| {
                let sa = children[a].1.unwrap_or(Score::MIN);
                let sb = children[b].1.unwrap_or(Score::MIN);
                sb.cmp(&sa)
            });
            let new_moves: Vec<ChessMove> = order.iter().map(|&i| moves[i]).collect();
            let new_children: Vec<(Option<Board>, Option<Score>)> =
                order.iter().map(|&i| children[i].clone()).collect();
            moves = new_moves;
            children = new_children;
            gpu_children = Some(children);
        }

        let mut best_score = -200000;

        for i in 0..moves.len() {
            if self.must_abort() {
                break;
            }

            let mv = moves[i];
            let hint_candidate: Option<Score>;

            let next_board: Option<Board> = match &gpu_children {
                Some(children) => match &children[i] {
                    (Some(b), Some(s)) => {
                        hint_candidate = Some(*s);
                        Some(b.clone())
                    }
                    _ => continue,
                },
                None => {
                    let mut b = board.clone();
                    if b.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                        continue;
                    }
                    hint_candidate = None;
                    Some(b)
                }
            };

            let mut nb = next_board.expect("child board made");

            let next_acc = match (accumulator, evaluator.nnue_network.as_ref()) {
                (Some(acc), Some(net)) => Some(acc.update_move(board, &mv, &net.weights)),
                _ => None,
            };

            let hint = if depth - 1 == 0 {
                if hint_candidate.is_some() {
                    self.hints_used += 1;
                }
                hint_candidate
            } else {
                None
            };

            let score = -self.search_with_eval(
                &mut nb,
                depth - 1,
                -beta,
                -alpha,
                tt,
                orderer,
                evaluator,
                next_acc.as_ref(),
                hint,
            );

            if self.is_aborted {
                return 0;
            }

            best_score = best_score.max(score);

            if score > alpha {
                alpha = score;
                if is_root {
                    self.best_move_at_root = Some(mv);
                }
            }

            if alpha >= beta {
                self.cutoffs += 1;
                orderer.record_killer(depth, mv);
                break;
            }
        }

        if !self.is_aborted {
            tt.store(board, depth, best_score, BoundType::Exact);
        }
        best_score
    }

    fn quiescence(
        &mut self,
        board: &Board,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        _orderer: &MoveOrderer,
        evaluator: &Evaluator,
        accumulator: Option<&NNUEAccumulator>,
        stand_pat_hint: Option<Score>,
    ) -> Score {
        if self.must_abort() {
            return 0;
        }

        if let Some(limit) = self.node_limit {
            if self.total_nodes() >= limit {
                self.is_aborted = true;
                return 0;
            }
        }

        self.qnodes += 1;

        let static_eval = match stand_pat_hint {
            Some(s) => s,
            None => evaluator.evaluate_with_accumulator(board, accumulator),
        };
        
        if static_eval >= beta {
            return beta;
        }

        alpha = alpha.max(static_eval);

        let moves = board.generate_moves();
        let mut best_score = static_eval;

        for mv in moves {
            if self.must_abort() {
                break;
            }

            let target = board.squares[mv.to.0][mv.to.1];
            if target == crate::board::pieces::Piece::Empty {
                continue;
            }

            let mut board_copy = board.clone();
            if board_copy.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                continue;
            }

            let next_acc = match (accumulator, evaluator.nnue_network.as_ref()) {
                (Some(acc), Some(net)) => Some(acc.update_move(board, &mv, &net.weights)),
                _ => None,
            };

            let score = -self.quiescence(
                &board_copy,
                -beta,
                -alpha,
                tt,
                _orderer,
                evaluator,
                next_acc.as_ref(),
                None,
            );

            if self.must_abort() {
                return 0;
            }

            best_score = best_score.max(score);
            alpha = alpha.max(score);

            if alpha >= beta {
                break;
            }
        }

        best_score
    }
}
