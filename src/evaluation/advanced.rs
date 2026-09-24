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
        let white_king = Self::find_king(board, Color::White);
        let black_king = Self::find_king(board, Color::Black);

        let white_safety = match white_king {
            Some(k) => Self::evaluate_king_safety(board, Color::White, k),
            None => 0,
        };
        let black_safety = match black_king {
            Some(k) => Self::evaluate_king_safety(board, Color::Black, k),
            None => 0,
        };

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

    fn evaluate_king_safety(board: &Board, color: Color, king_pos: (usize, usize)) -> Score {
        let mut score = 0;

        score += Self::pawn_shield(board, king_pos, color);
        score += Self::open_files_penalty(board, king_pos, color);
        score += Self::attack_danger(board, king_pos, color);
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

    /// Attack-based king danger. Sums weighted enemy attacks on the king and its
    /// 3x3 escape ring, then subtracts the pawn shield and nearby defenders.
    /// Also folds in: open files/diagonals (attacks require them), queen and rook
    /// proximity, weak escape squares, and an imminent-check penalty.
    fn attack_danger(board: &Board, king_pos: (usize, usize), color: Color) -> Score {
        let enemy = match color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        let (kr, kf) = (king_pos.0 as i32, king_pos.1 as i32);
        let mut targets = [(0usize, 0usize); 9];
        let mut n_targets = 0usize;
        targets[n_targets] = king_pos;
        n_targets += 1;
        for dr in -1i32..=1 {
            for df in -1i32..=1 {
                if dr == 0 && df == 0 {
                    continue;
                }
                let r = kr + dr;
                let f = kf + df;
                if r >= 0 && r < 8 && f >= 0 && f < 8 {
                    targets[n_targets] = (r as usize, f as usize);
                    n_targets += 1;
                }
            }
        }

        // attacked[i] == square attackable by an enemy piece this very eval.
        let mut attacked = [false; 9];
        let mut attack_units: Score = 0;
        let mut checking = false;
        let mut queen_proximity = false;
        let mut rook_proximity = false;

        for rank in 0..8 {
            for file in 0..8 {
                let sq = board.squares[rank][file];
                if sq == Piece::Empty {
                    continue;
                }
                if is_white(sq) != (enemy == Color::White) {
                    continue;
                }

                let dr = rank as i32 - kr;
                let df = file as i32 - kf;
                let d = dr.abs().max(df.abs());

                match sq {
                    Piece::WhiteQueen | Piece::BlackQueen => {
                        if d <= 3 {
                            queen_proximity = true;
                        }
                    }
                    Piece::WhiteRook | Piece::BlackRook => {
                        if d <= 4 {
                            rook_proximity = true;
                        }
                    }
                    _ => {}
                }

                let weight: Score = match sq {
                    Piece::WhitePawn | Piece::BlackPawn => 2,
                    Piece::WhiteKnight | Piece::BlackKnight => 4,
                    Piece::WhiteBishop | Piece::BlackBishop => 4,
                    Piece::WhiteRook | Piece::BlackRook => 5,
                    Piece::WhiteQueen | Piece::BlackQueen => 8,
                    Piece::WhiteKing | Piece::BlackKing => 1,
                    Piece::Empty => 0,
                };

                for (idx, target) in targets.iter().enumerate().take(n_targets) {
                    if Self::attacks_square(board, (rank, file), sq, *target) {
                        attacked[idx] = true;
                        let mult = if idx == 0 { 2 } else { 1 };
                        attack_units += weight * mult;
                        if idx == 0 {
                            checking = true;
                        }
                    }
                }
            }
        }

        // Defensive resources: pawn shield + friendly pieces camping near the king.
        let mut guard: Score = Self::pawn_shield(board, king_pos, color);
        let (lo_r, hi_r) = ((kr - 2).max(0), (kr + 2).min(7));
        let (lo_f, hi_f) = ((kf - 2).max(0), (kf + 2).min(7));
        for rank in lo_r..=hi_r {
            for file in lo_f..=hi_f {
                let sq = board.squares[rank as usize][file as usize];
                if sq == Piece::Empty {
                    continue;
                }
                if is_white(sq) != (color == Color::White) {
                    continue;
                }
                let d = (rank - kr).abs().max((file - kf).abs());
                if d > 0 && d <= 2 && sq != Self::color_king(color) {
                    guard += 6;
                }
            }
        }
        guard = guard.min(80);

        let mut danger = (attack_units - guard).max(0);
        danger = danger.min(120);

        if queen_proximity {
            danger = (danger + 6).min(120);
        }
        if rook_proximity {
            danger = (danger + 5).min(120);
        }
        if checking {
            danger = (danger + 15).min(120);
        }
        if safe_escape_squares(board, targets, n_targets, &attacked) == 0 {
            danger = (danger + 35).min(120);
        }

        -danger
    }

    fn color_king(color: Color) -> Piece {
        match color {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        }
    }

    #[cfg(test)]
    fn has_no_escape_squares(board: &Board, king_pos: (usize, usize), color: Color) -> bool {
        let enemy = match color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        let (kr, kf) = (king_pos.0 as i32, king_pos.1 as i32);
        let mut targets = [(0usize, 0usize); 9];
        let mut n_targets = 0usize;
        targets[n_targets] = king_pos;
        n_targets += 1;
        for dr in -1i32..=1 {
            for df in -1i32..=1 {
                if dr == 0 && df == 0 {
                    continue;
                }
                let r = kr + dr;
                let f = kf + df;
                if r >= 0 && r < 8 && f >= 0 && f < 8 {
                    targets[n_targets] = (r as usize, f as usize);
                    n_targets += 1;
                }
            }
        }

        let mut attacked = [false; 9];
        for rank in 0..8 {
            for file in 0..8 {
                let sq = board.squares[rank][file];
                if sq == Piece::Empty || is_white(sq) != (enemy == Color::White) {
                    continue;
                }
                for (idx, target) in targets.iter().enumerate().take(n_targets) {
                    if Self::attacks_square(board, (rank, file), sq, *target) {
                        attacked[idx] = true;
                    }
                }
            }
        }

        safe_escape_squares(board, targets, n_targets, &attacked) == 0
    }

    /// Cheap geometric attack test for `from`-piece vs `target`, independent of
    /// whose move it is. Sliding attacks require a clear line (empty path).
    fn attacks_square(
        board: &Board,
        from: (usize, usize),
        piece: Piece,
        target: (usize, usize),
    ) -> bool {
        let dr = target.0 as i32 - from.0 as i32;
        let df = target.1 as i32 - from.1 as i32;
        let ar = dr.abs();
        let af = df.abs();

        match piece {
            Piece::WhitePawn | Piece::BlackPawn => {
                if ar != 1 || af != 1 {
                    return false;
                }
                match piece {
                    Piece::WhitePawn => dr == -1,
                    Piece::BlackPawn => dr == 1,
                    _ => false,
                }
            }
            Piece::WhiteKnight | Piece::BlackKnight => {
                (ar == 2 && af == 1) || (ar == 1 && af == 2)
            }
            Piece::WhiteBishop | Piece::BlackBishop => {
                ar == af && ar != 0 && Self::path_clear(board, from, target)
            }
            Piece::WhiteRook | Piece::BlackRook => {
                (ar == 0 || af == 0) && ar + af > 0 && Self::path_clear(board, from, target)
            }
            Piece::WhiteQueen | Piece::BlackQueen => {
                ((ar == af && ar != 0) || (ar == 0 || af == 0)) && ar + af > 0
                    && Self::path_clear(board, from, target)
            }
            Piece::WhiteKing | Piece::BlackKing => dr.abs() <= 1 && df.abs() <= 1 && (dr != 0 || df != 0),
            Piece::Empty => false,
        }
    }

    fn path_clear(board: &Board, from: (usize, usize), target: (usize, usize)) -> bool {
        let dr = (target.0 as i32 - from.0 as i32).signum();
        let df = (target.1 as i32 - from.1 as i32).signum();
        let mut r = from.0 as i32 + dr;
        let mut f = from.1 as i32 + df;
        while (r, f) != (target.0 as i32, target.1 as i32) {
            if board.squares[r as usize][f as usize] != Piece::Empty {
                return false;
            }
            r += dr;
            f += df;
        }
        true
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
                return -20;
            }
            return 0;
        }

        if rights {
            return -(12 * dist_from_back).min(40);
        }
        -(18 * dist_from_back).min(70)
    }
}

fn safe_escape_squares(
    board: &Board,
    targets: [(usize, usize); 9],
    n_targets: usize,
    attacked: &[bool; 9],
) -> i32 {
    let mut safe = 0;
    for (idx, t) in targets.iter().enumerate().take(n_targets).skip(1) {
        if board.squares[t.0][t.1] == Piece::Empty && !attacked[idx] {
            safe += 1;
        }
    }
    safe
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

struct PawnFileStats {
    white_count: [u8; 8],
    black_count: [u8; 8],
    white_max_rank: [i8; 8],
    black_min_rank: [i8; 8],
}

impl PawnFileStats {
    fn collect(board: &Board) -> Self {
        let mut stats = PawnFileStats {
            white_count: [0; 8],
            black_count: [0; 8],
            white_max_rank: [-1; 8],
            black_min_rank: [8; 8],
        };
        for rank in 0..8 {
            for file in 0..8 {
                match board.squares[rank][file] {
                    Piece::WhitePawn => {
                        stats.white_count[file] += 1;
                        if stats.white_max_rank[file] < rank as i8 {
                            stats.white_max_rank[file] = rank as i8;
                        }
                    }
                    Piece::BlackPawn => {
                        stats.black_count[file] += 1;
                        if stats.black_min_rank[file] > rank as i8 {
                            stats.black_min_rank[file] = rank as i8;
                        }
                    }
                    _ => {}
                }
            }
        }
        stats
    }
}

pub struct PawnStructure;

impl PawnStructure {
    pub fn evaluate(board: &Board) -> Score {
        let stats = PawnFileStats::collect(board);
        let white_king = Self::find_king(board, Color::White);
        let black_king = Self::find_king(board, Color::Black);

        let white_score = Self::evaluate_pawns(board, Piece::WhitePawn, &stats, white_king, black_king);
        let black_score = Self::evaluate_pawns(board, Piece::BlackPawn, &stats, white_king, black_king);

        side_to_move_perspective(board, white_score - black_score)
    }

    fn evaluate_pawns(
        board: &Board,
        pawn_piece: Piece,
        stats: &PawnFileStats,
        white_king: Option<(usize, usize)>,
        black_king: Option<(usize, usize)>,
    ) -> Score {
        let mut score = 0;

        for rank in 0..8 {
            for file in 0..8 {
                if board.squares[rank][file] == pawn_piece {
                    if Self::is_isolated_pawn((rank, file), pawn_piece, stats) {
                        score -= 20;
                    }

                    if Self::is_doubled_pawn((rank, file), pawn_piece, stats) {
                        score -= 15;
                    }

                    if Self::is_passed_pawn((rank, file), pawn_piece, stats) {
                        score += Self::passed_pawn_bonus_with_kings(
                            board,
                            (rank, file),
                            pawn_piece,
                            white_king,
                            black_king,
                        );
                    }

                    if Self::is_backward_pawn(board, (rank, file), pawn_piece) {
                        score -= 10;
                    }
                }
            }
        }

        score
    }

    fn is_isolated_pawn(
        pos: (usize, usize),
        pawn_piece: Piece,
        stats: &PawnFileStats,
    ) -> bool {
        let counts = match pawn_piece {
            Piece::WhitePawn => &stats.white_count,
            Piece::BlackPawn => &stats.black_count,
            _ => panic!("is_isolated_pawn called with non-pawn"),
        };
        for file_opt in [
            if pos.1 > 0 { Some(pos.1 - 1) } else { None },
            if pos.1 < 7 { Some(pos.1 + 1) } else { None },
        ] {
            if let Some(file) = file_opt {
                if counts[file] > 0 {
                    return false;
                }
            }
        }

        true
    }

    fn is_doubled_pawn(
        pos: (usize, usize),
        pawn_piece: Piece,
        stats: &PawnFileStats,
    ) -> bool {
        let count = match pawn_piece {
            Piece::WhitePawn => stats.white_count[pos.1],
            Piece::BlackPawn => stats.black_count[pos.1],
            _ => panic!("is_doubled_pawn called with non-pawn"),
        };
        count > 1
    }

    fn is_passed_pawn(
        pos: (usize, usize),
        pawn_piece: Piece,
        stats: &PawnFileStats,
    ) -> bool {
        let file0 = pos.1 as i32;
        match pawn_piece {
            Piece::WhitePawn => {
                for f in (file0 - 1)..=(file0 + 1) {
                    if f < 0 || f >= 8 {
                        continue;
                    }
                    if stats.black_min_rank[f as usize] <= pos.0 as i8 {
                        return false;
                    }
                }
            }
            Piece::BlackPawn => {
                for f in (file0 - 1)..=(file0 + 1) {
                    if f < 0 || f >= 8 {
                        continue;
                    }
                    if stats.white_max_rank[f as usize] >= pos.0 as i8 {
                        return false;
                    }
                }
            }
            _ => panic!("is_passed_pawn called with non-pawn"),
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

    #[cfg(test)]
    fn passed_pawn_bonus(board: &Board, pos: (usize, usize), pawn_piece: Piece) -> Score {
        let white_king = Self::find_king(board, Color::White);
        let black_king = Self::find_king(board, Color::Black);
        Self::passed_pawn_bonus_with_kings(board, pos, pawn_piece, white_king, black_king)
    }

    fn passed_pawn_bonus_with_kings(
        board: &Board,
        pos: (usize, usize),
        pawn_piece: Piece,
        white_king: Option<(usize, usize)>,
        black_king: Option<(usize, usize)>,
    ) -> Score {
        let is_white = matches!(pawn_piece, Piece::WhitePawn);
        let (rank, file) = pos;
        let dist: i32 = if is_white {
            rank as i32
        } else {
            7 - rank as i32
        };
        if dist <= 0 {
            return 0;
        }

        let base = match dist {
            6 => 45,
            5 => 80,
            4 => 140,
            3 => 230,
            2 => 360,
            _ => 520,
        };

        let mut bonus = base;

        if file == 0 || file == 7 {
            bonus -= 25;
        }

        let promo_sq: (usize, usize) = if is_white { (0, file) } else { (7, file) };
        let (enemy_king, own_king) = if is_white {
            (black_king, white_king)
        } else {
            (white_king, black_king)
        };

        if let Some(enemy_king) = enemy_king {
            let kdist = Self::chebyshev(enemy_king, promo_sq);
            if kdist < 4 {
                bonus += (4 - kdist) * 30;
            }
        }

        if let Some(own_king) = own_king {
            let kdist = Self::chebyshev(own_king, pos);
            if kdist <= 2 {
                bonus += 40 + (2 - kdist) * 15;
            }
        }

        if is_white {
            for rr in (rank + 1)..8 {
                let sq = board.squares[rr][file];
                if sq != Piece::Empty {
                    if sq == Piece::WhiteRook || sq == Piece::WhiteQueen {
                        bonus += 30;
                    }
                    break;
                }
            }
        } else {
            for rr in (0..rank).rev() {
                let sq = board.squares[rr][file];
                if sq != Piece::Empty {
                    if sq == Piece::BlackRook || sq == Piece::BlackQueen {
                        bonus += 30;
                    }
                    break;
                }
            }
        }

        if dist == 1 {
            if let Some(enemy_king) = enemy_king {
                if Self::queen_attacks(board, promo_sq, enemy_king, pos) {
                    bonus += 80;
                }
            }
        }

        bonus
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

    fn chebyshev(a: (usize, usize), b: (usize, usize)) -> i32 {
        (a.0 as i32 - b.0 as i32)
            .abs()
            .max((a.1 as i32 - b.1 as i32).abs())
    }

    fn queen_attacks(
        board: &Board,
        from: (usize, usize),
        target: (usize, usize),
        ignore: (usize, usize),
    ) -> bool {
        let dr = target.0 as i32 - from.0 as i32;
        let df = target.1 as i32 - from.1 as i32;
        if dr == 0 && df == 0 {
            return false;
        }
        if dr != 0 && df != 0 && dr.abs() != df.abs() {
            return false;
        }
        let r_step = dr.signum();
        let f_step = df.signum();
        let mut r = from.0 as i32 + r_step;
        let mut f = from.1 as i32 + f_step;
        while (r, f) != (target.0 as i32, target.1 as i32) {
            if board.squares[r as usize][f as usize] != Piece::Empty
                && (r as usize, f as usize) != ignore
            {
                return false;
            }
            r += r_step;
            f += f_step;
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
        let white = KingSafety::evaluate_king_safety(
            &board,
            Color::White,
            KingSafety::find_king(&board, Color::White).unwrap(),
        );
        let black = KingSafety::evaluate_king_safety(
            &board,
            Color::Black,
            KingSafety::find_king(&board, Color::Black).unwrap(),
        );
        assert!(white > black, "castled white king should be safer, w={} b={}", white, black);
    }

    #[test]
    fn test_passed_pawn_bonus_is_nonlinear_and_grows_toward_promotion() {
        // White pawn c7 (one square from queening) vs c5, same kings elsewhere.
        let near = Board::from_fen("4k3/2P5/8/8/8/8/8/7K w - - 0 1").unwrap();
        let far = Board::from_fen("4k3/8/8/2P5/8/8/8/7K w - - 0 1").unwrap();
        let bonus_near = PawnStructure::passed_pawn_bonus(&near, (1, 2), Piece::WhitePawn);
        let bonus_far = PawnStructure::passed_pawn_bonus(&far, (3, 2), Piece::WhitePawn);
        assert!(
            bonus_near > bonus_far * 2,
            "expected strongly nonlinear growth: c7={} should dominate c5={}",
            bonus_near,
            bonus_far
        );
    }

    #[test]
    fn test_passed_pawn_promotion_with_check_bonus() {
        // c7 passer: promoting to a queen on c8 gives check against e8 (diagonal/straight),
        // but not against a distant king — the check bonus should make the first higher.
        let with_check = Board::from_fen("4k3/2P5/8/8/8/8/8/7K w - - 0 1").unwrap();
        let no_check = Board::from_fen("8/2P5/8/3k4/8/8/8/7K w - - 0 1").unwrap();
        let bonus_check = PawnStructure::passed_pawn_bonus(&with_check, (1, 2), Piece::WhitePawn);
        let bonus_no_check = PawnStructure::passed_pawn_bonus(&no_check, (1, 2), Piece::WhitePawn);
        assert!(
            bonus_check > bonus_no_check,
            "promotion with check should score higher: {} vs {}",
            bonus_check,
            bonus_no_check
        );
    }

    #[test]
    fn test_edge_passed_pawn_reduced() {
        // Identical context, but edge (a-pawn) vs central (c-pawn) passer.
        let edge = Board::from_fen("8/8/8/P2k4/8/8/8/7K w - - 0 1").unwrap();
        let central = Board::from_fen("8/8/8/2Pk4/8/8/8/7K w - - 0 1").unwrap();
        let bonus_edge = PawnStructure::passed_pawn_bonus(&edge, (3, 0), Piece::WhitePawn);
        let bonus_central = PawnStructure::passed_pawn_bonus(&central, (3, 2), Piece::WhitePawn);
        assert!(
            bonus_central > bonus_edge,
            "central passer should outscore edge passer: {} vs {}",
            bonus_central,
            bonus_edge
        );
    }

    #[test]
    fn test_king_danger_surge_with_queen_and_rook_assault() {
        // Exposed g1 king, black queen already attacking down the open g-file
        // and a black rook on the back rank: the mating structure the engine leaked.
        let attacked = Board::from_fen("4k3/8/8/8/8/8/6q1/5rk1 w - - 0 1").unwrap();
        // Sheltered g1 king with f/g/h pawns stacked on the 2nd/3rd ranks;
        // attackers are far away.
        let sheltered = Board::from_fen("r3k3/6q1/8/8/8/5PPP/5PPP/6K1 w - - 0 1").unwrap();
        let danger_attacked = KingSafety::attack_danger(&attacked, (7, 6), Color::White);
        let danger_sheltered = KingSafety::attack_danger(&sheltered, (7, 6), Color::White);
        assert!(
            danger_attacked < danger_sheltered,
            "exposed king should be in grave danger: {} vs {}",
            danger_attacked,
            danger_sheltered
        );
        assert!(danger_attacked < -80, "expected heavy penalty, got {}", danger_attacked);
    }

    #[test]
    fn test_queen_proximity_deepens_danger() {
        // Black queen on g4 closes in on an unshielded g1 king across clear ranks.
        let queen_storm = Board::from_fen("4k3/8/8/8/6q1/8/8/6K1 w - - 0 1").unwrap();
        // Same, but the queen sits on the far queenside - no immediate threat.
        let safe = Board::from_fen("4k3/8/8/8/q7/8/8/6K1 w - - 0 1").unwrap();
        let danger_storm = KingSafety::attack_danger(&queen_storm, (7, 6), Color::White);
        let danger_safe = KingSafety::attack_danger(&safe, (7, 6), Color::White);
        assert!(
            danger_storm < danger_safe,
            "queen proximity should raise danger: {} vs {}",
            danger_storm,
            danger_safe
        );
    }

    #[test]
    fn test_trapped_king_no_escape_squares() {
        // White king h1 with all three retreat squares attacked.
        let trapped = Board::from_fen("4k3/8/8/8/8/8/4q3/5r1K w - - 0 1").unwrap();
        assert!(
            KingSafety::has_no_escape_squares(&trapped, (7, 7), Color::White),
            "king h1 should have no safe retreat"
        );
    }
}