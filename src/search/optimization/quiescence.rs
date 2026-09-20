use crate::board::Board;
use crate::board::pieces::Piece;
use crate::evaluation::Score;
use crate::transposition_table::TranspositionTable;

pub struct QuiescenceSearch;

impl QuiescenceSearch {
    pub fn search(
        board: &Board,
        mut alpha: Score,
        beta: Score,
        tt: &mut TranspositionTable,
    ) -> Score {
        let static_eval = crate::evaluation::Evaluator::hand_crafted_evaluate(board);
        
        if static_eval >= beta {
            return beta;
        }

        if static_eval > alpha {
            alpha = static_eval;
        }

        let mut moves = board.generate_moves();
        
        moves.retain(|mv| {
            board.squares[mv.to.0][mv.to.1] != Piece::Empty
        });

        let mut best_score = static_eval;

        for mv in moves {
            let mut board_copy = board.clone();
            if board_copy.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                continue;
            }

            let score = -Self::search(&board_copy, -beta, -alpha, tt);
            best_score = best_score.max(score);
            alpha = alpha.max(score);

            if alpha >= beta {
                break;
            }
        }

        best_score
    }

    pub fn is_quiet(board: &Board, last_move: crate::board::ChessMove) -> bool {
        let target = board.squares[last_move.to.0][last_move.to.1];
        target == Piece::Empty
    }
}
