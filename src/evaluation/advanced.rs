use crate::board::{Board, Color};
use crate::board::pieces::{is_black, is_white, same_color, Piece};
use crate::evaluation::material::MaterialEvaluation;
use crate::evaluation::Score;

fn side_to_move_perspective(board: &Board, white_minus_black: Score) -> Score {
    match board.turn {
        Color::White => white_minus_black,
        Color::Black => -white_minus_black,
    }
}

const KNIGHT_OFFSETS: [(i32, i32); 8] = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
];

pub struct PieceMobility;

impl PieceMobility {
    pub fn evaluate(board: &Board) -> Score {
        let mut white_mobility = 0;
        let mut black_mobility = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = board.squares[rank][file];
                if piece == Piece::Empty {
                    continue;
                }

                let count = Self::mobility_moves(board, (rank, file), piece) as Score;

                if is_white(piece) {
                    white_mobility += count;
                } else if is_black(piece) {
                    black_mobility += count;
                }
            }
        }

        side_to_move_perspective(board, (white_mobility - black_mobility) * 2)
    }

    fn mobility_moves(board: &Board, from: (usize, usize), piece: Piece) -> usize {
        match piece {
            Piece::WhiteKnight | Piece::BlackKnight => {
                let mut count = 0;
                for (dr, df) in KNIGHT_OFFSETS {
                    let r = from.0 as i32 + dr;
                    let f = from.1 as i32 + df;
                    if r >= 0 && r < 8 && f >= 0 && f < 8
                        && !same_color(board.squares[r as usize][f as usize], piece)
                    {
                        count += 1;
                    }
                }
                count
            }
            Piece::WhiteBishop | Piece::BlackBishop => {
                Self::ray_moves(board, from, piece, &[(-1, -1), (-1, 1), (1, -1), (1, 1)])
            }
            Piece::WhiteRook | Piece::BlackRook => {
                Self::ray_moves(board, from, piece, &[(-1, 0), (1, 0), (0, -1), (0, 1)])
            }
            Piece::WhiteQueen | Piece::BlackQueen => Self::ray_moves(
                board,
                from,
                piece,
                &[
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ],
            ),
            _ => 0,
        }
    }

    fn ray_moves(
        board: &Board,
        from: (usize, usize),
        piece: Piece,
        dirs: &[(i32, i32)],
    ) -> usize {
        let mut count = 0;
        for (dr, df) in dirs {
            let mut r = from.0 as i32 + dr;
            let mut f = from.1 as i32 + df;
            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                let sq = board.squares[r as usize][f as usize];
                if sq == Piece::Empty {
                    count += 1;
                } else {
                    if !same_color(sq, piece) {
                        count += 1;
                    }
                    break;
                }
                r += dr;
                f += df;
            }
        }
        count
    }
}

pub struct KingSafety;

impl KingSafety {
    pub fn evaluate(board: &Board) -> Score {
        let white_safety = Self::evaluate_king_safety(board, Color::White);
        let black_safety = Self::evaluate_king_safety(board, Color::Black);

        side_to_move_perspective(board, white_safety - black_safety)
    }

    fn find_king(board: &Board, color: Color) -> Option<(usize, usize)> {
        let king = match color {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        };
        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == king {
                    return Some((rank, file));
                }
            }
        }
        None
    }

    fn evaluate_king_safety(board: &Board, color: Color) -> Score {
        let Some(king_pos) = Self::find_king(board, color) else {
            return 0;
        };

        let mut score = 0;

        score += Self::pawn_shield(board, king_pos, color);
        score += Self::open_files_penalty(board, king_pos, color);
        score -= Self::king_danger(board, king_pos, color);
        score += Self::exposed_king_penalty(board, king_pos, color);

        score
    }

    fn pawn_shield(board: &Board, king_pos: (usize, usize), color: Color) -> Score {
        let pawn = match color {
            Color::White => Piece::WhitePawn,
            Color::Black => Piece::BlackPawn,
        };
        let (kr, kf) = king_pos;
        let sign: i32 = match color {
            Color::White => -1,
            Color::Black => 1,
        };

        let mut score = 0;
        for ring in 1..=2 {
            for df in -1..=1 {
                let r = kr as i32 + sign * ring;
                let f = kf as i32 + df;
                if r >= 0 && r < 8 && f >= 0 && f < 8 {
                    if board.squares[r as usize][f as usize] == pawn {
                        score += if ring == 1 { 15 } else { 8 };
                    }
                }
            }
        }
        score.min(46)
    }

    fn open_files_penalty(board: &Board, king_pos: (usize, usize), color: Color) -> Score {
        let pawn = match color {
            Color::White => Piece::WhitePawn,
            Color::Black => Piece::BlackPawn,
        };
        let (_, kf) = king_pos;

        let mut penalty = 0;
        for df in -1i32..=1 {
            let f = kf as i32 + df;
            if f < 0 || f >= 8 {
                continue;
            }
            let mut has_own_pawn = false;
            for rank in 0..8 {
                if board.squares[rank][f as usize] == pawn {
                    has_own_pawn = true;
                    break;
                }
            }
            if !has_own_pawn {
                penalty += if df == 0 { 12 } else { 7 };
            }
        }
        -penalty
    }

    fn king_danger(board: &Board, king_pos: (usize, usize), color: Color) -> Score {
        let (kr, kf) = (king_pos.0 as i32, king_pos.1 as i32);
        let mut danger: Score = 0;
        let mut defenders: Score = 0;

        for rank in 0..8 {
            for file in 0..8 {
                let sq = board.squares[rank][file];
                if sq == Piece::Empty {
                    continue;
                }

                let d = (rank as i32 - kr).abs().max((file as i32 - kf).abs());
                if d == 0 {
                    continue;
                }

                if is_white(sq) == (color == Color::White) {
                    if d <= 2 {
                        defenders += 4 - (d - 1) * 2;
                    }
                    continue;
                }

                danger += match sq {
                    Piece::WhitePawn | Piece::BlackPawn => {
                        if d <= 2 {
                            6 * (3 - d)
                        } else {
                            0
                        }
                    }
                    Piece::WhiteKnight | Piece::BlackKnight => {
                        if d <= 3 {
                            6 * (4 - d)
                        } else {
                            0
                        }
                    }
                    Piece::WhiteBishop | Piece::BlackBishop => {
                        if d <= 3 {
                            4 * (4 - d)
                        } else {
                            0
                        }
                    }
                    Piece::WhiteRook | Piece::BlackRook => {
                        if d <= 4 {
                            5 * (5 - d)
                        } else {
                            0
                        }
                    }
                    Piece::WhiteQueen | Piece::BlackQueen => {
                        if d <= 4 {
                            8 * (5 - d)
                        } else {
                            0
                        }
                    }
                    Piece::WhiteKing | Piece::BlackKing => 0,
                    Piece::Empty => 0,
                };
            }
        }

        danger = danger.saturating_sub(defenders);
        danger.clamp(0, 90)
    }

    fn exposed_king_penalty(board: &Board, king_pos: (usize, usize), color: Color) -> Score {
        let (white_mat, black_mat) = MaterialEvaluation::material_count(board);
        if white_mat + black_mat < 2600 {
            return 0;
        }

        let (kr, kf) = king_pos;
        let dist_from_back: i32 = match color {
            Color::White => 7 - kr as i32,
            Color::Black => kr as i32,
        };

        let rights = match color {
            Color::White => {
                board.castling_rights.white_kingside || board.castling_rights.white_queenside
            }
            Color::Black => {
                board.castling_rights.black_kingside || board.castling_rights.black_queenside
            }
        };

        if dist_from_back == 0 {
            if !rights && (kf == 3 || kf == 4) {
                return -40;
            }
            return 0;
        }

        if rights {
            return -(25 * dist_from_back).min(75);
        }
        -(35 * dist_from_back).min(120)
    }
}

pub struct RookActivity;

impl RookActivity {
    pub fn evaluate(board: &Board) -> Score {
        let white = Self::rook_score(board, Color::White);
        let black = Self::rook_score(board, Color::Black);

        side_to_move_perspective(board, white - black)
    }

    fn rook_score(board: &Board, color: Color) -> Score {
        let rook = match color {
            Color::White => Piece::WhiteRook,
            Color::Black => Piece::BlackRook,
        };

        let mut rooks = Vec::new();
        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == rook {
                    rooks.push((rank, file));
                }
            }
        }

        let mut score = 0;
        for (r, f) in &rooks {
            let (own_pawns, any_pawns) = Self::file_pawns(board, *f as i32, color);
            if !any_pawns {
                score += 30;
            } else if !own_pawns {
                score += 16;
            }

            match color {
                Color::White => {
                    if *r == 1 {
                        score += 35;
                    } else if *r == 0 {
                        score += 12;
                    }
                }
                Color::Black => {
                    if *r == 6 {
                        score += 35;
                    } else if *r == 7 {
                        score += 12;
                    }
                }
            }
        }

        for i in 0..rooks.len() {
            for j in (i + 1)..rooks.len() {
                if Self::clear_path(board, rooks[i], rooks[j]) {
                    score += 12;
                }
            }
        }

        score
    }

    fn file_pawns(board: &Board, file: i32, color: Color) -> (bool, bool) {
        let pawn = match color {
            Color::White => Piece::WhitePawn,
            Color::Black => Piece::BlackPawn,
        };
        let opponent_pawn = match color {
            Color::White => Piece::BlackPawn,
            Color::Black => Piece::WhitePawn,
        };

        let mut own = false;
        let mut any = false;
        for rank in 0..8 {
            let sq = board.squares[rank][file as usize];
            if sq == pawn {
                own = true;
                any = true;
            } else if sq == opponent_pawn {
                any = true;
            }
        }
        (own, any)
    }

    fn clear_path(board: &Board, a: (usize, usize), b: (usize, usize)) -> bool {
        if a.0 != b.0 && a.1 != b.1 {
            return false;
        }
        let dr = (b.0 as i32 - a.0 as i32).signum();
        let df = (b.1 as i32 - a.1 as i32).signum();
        let mut r = a.0 as i32 + dr;
        let mut f = a.1 as i32 + df;
        while (r, f) != (b.0 as i32, b.1 as i32) {
            if board.squares[r as usize][f as usize] != Piece::Empty {
                return false;
            }
            r += dr;
            f += df;
        }
        true
    }
}

pub struct PawnStructure;

impl PawnStructure {
    pub fn evaluate(board: &Board) -> Score {
        let white_score = Self::evaluate_pawns(board, Piece::WhitePawn);
        let black_score = Self::evaluate_pawns(board, Piece::BlackPawn);

        side_to_move_perspective(board, white_score - black_score)
    }

    fn evaluate_pawns(board: &Board, pawn_piece: Piece) -> Score {
        let mut score = 0;

        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == pawn_piece {
                    if Self::is_isolated_pawn(board, (rank, file), pawn_piece) {
                        score -= 20;
                    }

                    if Self::is_doubled_pawn(board, (rank, file), pawn_piece) {
                        score -= 15;
                    }

                    if Self::is_passed_pawn(board, (rank, file), pawn_piece) {
                        let is_white = matches!(pawn_piece, Piece::WhitePawn);
                        let dist_to_promotion = if is_white { rank } else { 7 - rank };
                        score += 25 + (7 - dist_to_promotion as Score) * 15;
                    }

                    if Self::is_backward_pawn(board, (rank, file), pawn_piece) {
                        score -= 10;
                    }
                }
            }
        }

        score
    }

    fn is_isolated_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        let adjacent_files = [
            if pos.1 > 0 { Some(pos.1 - 1) } else { None },
            if pos.1 < 7 { Some(pos.1 + 1) } else { None },
        ];

        for file_opt in adjacent_files.iter() {
            if let Some(file) = file_opt {
                for rank in 0..8 {
                    if board.squares[rank][*file] == pawn_piece {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn is_doubled_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        for rank in 0..8 {
            if rank != pos.0 && board.squares[rank][pos.1] == pawn_piece {
                return true;
            }
        }
        false
    }

    fn is_passed_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        let is_white = matches!(pawn_piece, Piece::WhitePawn);
        let opponent_pawn = if is_white { Piece::BlackPawn } else { Piece::WhitePawn };

        let (scan_start, scan_end) = if is_white {
            (0usize, pos.0)
        } else {
            (pos.0 + 1, 8usize)
        };

        for file_offset in -1..=1 {
            let file = pos.1 as i32 + file_offset;
            if file < 0 || file >= 8 {
                continue;
            }
            for rank in scan_start..scan_end {
                if board.squares[rank][file as usize] == opponent_pawn {
                    return false;
                }
            }
        }

        true
    }

    fn is_backward_pawn(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> bool {
        let is_white = matches!(pawn_piece, Piece::WhitePawn);
        let support_rank = if is_white {
            pos.0 + 1
        } else {
            if pos.0 > 0 {
                pos.0 - 1
            } else {
                return false;
            }
        };

        let support_files = [
            if pos.1 > 0 { Some(pos.1 - 1) } else { None },
            if pos.1 < 7 { Some(pos.1 + 1) } else { None },
        ];

        for file_opt in support_files.iter() {
            if let Some(file) = file_opt {
                if board.squares[support_rank][*file] == pawn_piece {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position_near_equal() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("Failed to parse FEN");
        let score = PieceMobility::evaluate(&board)
            + KingSafety::evaluate(&board)
            + RookActivity::evaluate(&board)
            + PawnStructure::evaluate(&board);
        assert!(score.abs() < 200, "startpos should be near equal, got {}", score);
    }

    #[test]
    fn test_mobility_perspective_flips_with_turn() {
        let fen_white = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1";
        let fen_black = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 0 1";
        let w = Board::from_fen(fen_white).unwrap();
        let b = Board::from_fen(fen_black).unwrap();
        assert_eq!(PieceMobility::evaluate(&w), -PieceMobility::evaluate(&b));
    }

    #[test]
    fn test_pawn_shield_rewards_sheltered_king() {
        // White king on g1 with f/g/h-pawns still on the 2nd/3rd ranks.
        let sheltered = "r3k2r/ppp2ppp/2n5/2pp4/2P5/2N2N2/PP2PPPP/R3K2R w KQkq - 0 1";
        let board = Board::from_fen(sheltered).unwrap();
        let white = KingSafety::evaluate_king_safety(&board, Color::White);
        let black = KingSafety::evaluate_king_safety(&board, Color::Black);
        assert!(white > black, "castled white king should be safer, w={} b={}", white, black);
    }
}