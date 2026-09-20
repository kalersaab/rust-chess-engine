use crate::board::Board;
use crate::evaluation::Score;
use crate::transposition_table::{TranspositionTable, BoundType};
use crate::move_ordering::MoveOrderer;

pub struct PrincipalVariationSearch {
    pub nodes: u64,
    pub qnodes: u64,
    pub cutoffs: u64,
    pub full_window_searches: u32,
}

impl PrincipalVariationSearch {
    pub fn new() -> Self {
        PrincipalVariationSearch {
            nodes: 0,
            qnodes: 0,
            cutoffs: 0,
            full_window_searches: 0,
        }
    }

    pub fn search(
        &mut self,
        board: &mut Board,
        depth: u32,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        orderer: &mut MoveOrderer,
    ) -> Score {
        if depth == 0 {
            return self.quiescence(board, alpha, beta, tt, orderer);
        }

        self.nodes += 1;

        let mut moves = board.generate_moves();
        
        if moves.is_empty() {
            if board.is_in_check(board.turn) {
                return -100000 + (20 - depth) as Score;
            } else {
                return 0;
            }
        }

        orderer.sort_moves(&mut moves, board, depth, None);

        let mut best_score = -200000;
        let mut first_move = true;

        for mv in moves {
            let from = Board::square_to_string(mv.from);
            let to = Board::square_to_string(mv.to);

            let mut next_board = board.clone();
            if next_board.make_move(&from, &to).is_err() {
                continue;
            }

            let score = if first_move {
                self.full_window_searches += 1;
                -self.search(&mut next_board, depth - 1, -beta, -alpha, tt, orderer)
            } else {
                let null_window_score = -self.search(&mut next_board, depth - 1, -alpha - 1, -alpha, tt, orderer);
                
                if null_window_score > alpha && null_window_score < beta {
                    self.full_window_searches += 1;
                    -self.search(&mut next_board, depth - 1, -beta, -null_window_score, tt, orderer)
                } else {
                    null_window_score
                }
            };

            best_score = best_score.max(score);
            alpha = alpha.max(score);

            first_move = false;

            if alpha >= beta {
                self.cutoffs += 1;
                orderer.record_killer(depth, mv);
                break;
            }
        }

        tt.store(board, depth, best_score, BoundType::Exact);
        best_score
    }

    fn quiescence(
        &mut self,
        board: &Board,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
        _orderer: &MoveOrderer,
    ) -> Score {
        self.qnodes += 1;

        let static_eval = crate::evaluation::Evaluator::evaluate(board);
        
        if static_eval >= beta {
            return beta;
        }

        alpha = alpha.max(static_eval);

        let moves = board.generate_moves();
        let mut best_score = static_eval;

        for mv in moves {
            let target = board.squares[mv.to.0][mv.to.1];
            if target == crate::board::pieces::Piece::Empty {
                continue;
            }

            let from = Board::square_to_string(mv.from);
            let to = Board::square_to_string(mv.to);

            let mut board_copy = board.clone();
            if board_copy.make_move(&from, &to).is_err() {
                continue;
            }

            let score = -self.quiescence(&board_copy, -beta, -alpha, tt, _orderer);
            best_score = best_score.max(score);
            alpha = alpha.max(score);

            if alpha >= beta {
                break;
            }
        }

        best_score
    }
}
