use crate::board::{Board, ChessMove};
use crate::board::pieces::Piece;
use crate::evaluation::Score;

pub struct StaticExchangeEvaluation;

impl StaticExchangeEvaluation {
    const PIECE_VALUES: [Score; 8] = [0, 1, 3, 3, 5, 9, 200, 0];

    pub fn evaluate(board: &Board, chess_move: ChessMove) -> Score {
        let target_piece = board.squares[chess_move.to.0][chess_move.to.1];
        
        if target_piece == Piece::Empty {
            return 0;
        }

        let mut board_copy = board.clone();
        
        let from_str = Board::square_to_string(chess_move.from);
        let to_str = Board::square_to_string(chess_move.to);
        
        if board_copy.make_move(&from_str, &to_str).is_err() {
            return 0;
        }

        let target_value = Self::piece_value(target_piece);
        let attacker_value = Self::piece_value(board.squares[chess_move.from.0][chess_move.from.1]);

        let recapture_score = Self::evaluate_recaptures(&board_copy, chess_move.to);

        target_value - recapture_score.min(attacker_value)
    }

    fn evaluate_recaptures(board: &Board, square: (usize, usize)) -> Score {
        let mut best_recapture = 0;

        let moves = board.generate_moves();
        
        for mv in moves {
            if mv.to == square {
                let captured = board.squares[square.0][square.1];
                if captured != Piece::Empty {
                    let value = Self::piece_value(captured);
                    best_recapture = best_recapture.max(value);
                }
            }
        }

        best_recapture
    }

    pub fn piece_value(piece: Piece) -> Score {
        match piece {
            Piece::WhitePawn | Piece::BlackPawn => Self::PIECE_VALUES[1],
            Piece::WhiteKnight | Piece::BlackKnight => Self::PIECE_VALUES[2],
            Piece::WhiteBishop | Piece::BlackBishop => Self::PIECE_VALUES[3],
            Piece::WhiteRook | Piece::BlackRook => Self::PIECE_VALUES[4],
            Piece::WhiteQueen | Piece::BlackQueen => Self::PIECE_VALUES[5],
            Piece::WhiteKing | Piece::BlackKing => Self::PIECE_VALUES[6],
            Piece::Empty => 0,
        }
    }

    pub fn is_good_capture(board: &Board, chess_move: ChessMove) -> bool {
        let target = board.squares[chess_move.to.0][chess_move.to.1];
        
        if target == Piece::Empty {
            return false;
        }

        let attacker = board.squares[chess_move.from.0][chess_move.from.1];
        let target_value = Self::piece_value(target);
        let attacker_value = Self::piece_value(attacker);

        target_value >= attacker_value
    }

    pub fn is_bad_capture(board: &Board, chess_move: ChessMove) -> bool {
        Self::evaluate(board, chess_move) < 0
    }
}
