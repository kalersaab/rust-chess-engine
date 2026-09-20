use super::board::{Board, Color, GameStatus};
use super::chess_move::ChessMove;
use super::pieces::*;

impl Board {
    fn generate_pseudo_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::new();

        for rank in 0..8 {
            for file in 0..8 {
                let piece = self.squares[rank][file];

                if piece == Piece::Empty {
                    continue;
                }

                match self.turn {
                    Color::White if !is_white(piece) => continue,
                    Color::Black if !is_black(piece) => continue,
                    _ => {}
                }
                
                match piece {
                    Piece::WhitePawn | Piece::BlackPawn => {
                        self.generate_pawn_moves(rank, file, &mut moves);
                    }

                    Piece::WhiteKnight | Piece::BlackKnight => {
                        self.generate_knight_moves(rank, file, &mut moves);
                    }

                    Piece::WhiteBishop | Piece::BlackBishop => {
                        self.generate_bishop_moves(rank, file, &mut moves);
                    }

                    Piece::WhiteRook | Piece::BlackRook => {
                        self.generate_rook_moves(rank, file, &mut moves);
                    }

                    Piece::WhiteQueen | Piece::BlackQueen => {
                        self.generate_bishop_moves(rank, file, &mut moves);
                        self.generate_rook_moves(rank, file, &mut moves);
                    }

                    Piece::WhiteKing | Piece::BlackKing => {
                        self.generate_king_moves(rank, file, &mut moves);
                    }

                    Piece::Empty => {}
                }
            }
        }

        moves
    }
    pub fn generate_moves(&self) -> Vec<ChessMove> {
        let pseudo_moves = self.generate_pseudo_moves();
        let mut legal_moves = Vec::new();

        for mv in pseudo_moves {
            let mut next_board = self.clone();

            if next_board.execute_move(mv.from, mv.to, mv.move_type).is_err() {
                continue;
            }

            let moving_color = self.turn;

            if !next_board.is_in_check(moving_color) {
                legal_moves.push(mv);
            }
        }

        legal_moves
    }
    pub fn is_checkmate(&self) -> bool {
        self.is_in_check(self.turn)
            && self.generate_moves().is_empty()
    }

    pub fn is_stalemate(&self) -> bool {
        !self.is_in_check(self.turn)
            && self.generate_moves().is_empty()
    }

    pub fn is_check(&self) -> bool {
        self.is_in_check(self.turn)
    }
    pub fn game_status(&self) -> GameStatus {
        let in_check = self.is_in_check(self.turn);
        let legal_moves = self.generate_moves();

        if legal_moves.is_empty() {
            if in_check {
                GameStatus::Checkmate
            } else {
                GameStatus::Stalemate
            }
        } else if in_check {
            GameStatus::Check
        } else {
            GameStatus::Playing
        }
    }

    pub fn print_game_status(&self) {
        match self.game_status() {
            GameStatus::Playing => {
                println!("Status: Playing");
            }

            GameStatus::Check => {
                println!("Status: Check");
            }

            GameStatus::Checkmate => {
                let winner = match self.turn {
                    Color::White => "Black",
                    Color::Black => "White",
                };

                println!("Status: Checkmate");
                println!("Winner: {}", winner);
            }

            GameStatus::Stalemate => {
                println!("Status: Stalemate");
            }
        }
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        let king_square = match self.find_king(color) {
            Some(square) => square,
            None => return false,
        };

        let attacking_color = match color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        self.is_square_attacked(king_square, attacking_color)
    }
    pub fn is_square_attacked(
        &self,
        target: (usize, usize),
        by_color: crate::board::Color,
    ) -> bool {
        let (_target_rank, _target_file) = target;

        for rank in 0..8 {
            for file in 0..8 {
                let piece = self.squares[rank][file];

                if piece == Piece::Empty {
                    continue;
                }

                let correct_color = match by_color {
                    Color::White => is_white(piece),
                    Color::Black => is_black(piece),
                };

                if !correct_color {
                    continue;
                }

                let from = (rank, file);

                match piece {
                    Piece::WhitePawn | Piece::BlackPawn => {
                        if self.pawn_attacks(from, target, by_color) {
                            return true;
                        }
                    }

                    Piece::WhiteKnight | Piece::BlackKnight => {
                        if self.knight_attacks(from, target) {
                            return true;
                        }
                    }

                    Piece::WhiteBishop | Piece::BlackBishop => {
                        if self.bishop_attacks(from, target) {
                            return true;
                        }
                    }

                    Piece::WhiteRook | Piece::BlackRook => {
                        if self.rook_attacks(from, target) {
                            return true;
                        }
                    }

                    Piece::WhiteQueen | Piece::BlackQueen => {
                        if self.bishop_attacks(from, target)
                            || self.rook_attacks(from, target)
                        {
                            return true;
                        }
                    }

                    Piece::WhiteKing | Piece::BlackKing => {
                        if self.king_attacks(from, target) {
                            return true;
                        }
                    }

                    Piece::Empty => {}
                }
            }
        }

        false
    }
    fn pawn_attacks(
        &self,
        from: (usize, usize),
        target: (usize, usize),
        color: Color,
    ) -> bool {
        let (rank, file) = from;
        let (target_rank, target_file) = target;

        let direction: i32 = match color {
            Color::White => -1,
            Color::Black => 1,
        };

        let rank = rank as i32;
        let file = file as i32;

        let target_rank = target_rank as i32;
        let target_file = target_file as i32;

        target_rank == rank + direction
            && (target_file == file - 1 || target_file == file + 1)
    }
    fn knight_attacks(
        &self,
        from: (usize, usize),
        target: (usize, usize),
    ) -> bool {
        let dr = (from.0 as i32 - target.0 as i32).abs();
        let df = (from.1 as i32 - target.1 as i32).abs();

        (dr == 2 && df == 1) || (dr == 1 && df == 2)
    }
    fn king_attacks(
        &self,
        from: (usize, usize),
        target: (usize, usize),
    ) -> bool {
        let dr = (from.0 as i32 - target.0 as i32).abs();
        let df = (from.1 as i32 - target.1 as i32).abs();

        dr <= 1 && df <= 1 && (dr != 0 || df != 0)
    }
    fn bishop_attacks(
        &self,
        from: (usize, usize),
        target: (usize, usize),
    ) -> bool {
        let dr = target.0 as i32 - from.0 as i32;
        let df = target.1 as i32 - from.1 as i32;

        if dr.abs() != df.abs() || dr == 0 {
            return false;
        }

        let rank_step = dr.signum();
        let file_step = df.signum();

        let mut rank = from.0 as i32 + rank_step;
        let mut file = from.1 as i32 + file_step;

        while (rank, file) != (target.0 as i32, target.1 as i32) {
            if self.squares[rank as usize][file as usize] != Piece::Empty {
                return false;
            }

            rank += rank_step;
            file += file_step;
        }

        true
    }
    fn rook_attacks(
        &self,
        from: (usize, usize),
        target: (usize, usize),
    ) -> bool {
        let same_rank = from.0 == target.0;
        let same_file = from.1 == target.1;

        if !same_rank && !same_file {
            return false;
        }

        let rank_step = (target.0 as i32 - from.0 as i32).signum();
        let file_step = (target.1 as i32 - from.1 as i32).signum();

        let mut rank = from.0 as i32 + rank_step;
        let mut file = from.1 as i32 + file_step;

        while (rank, file) != (target.0 as i32, target.1 as i32) {
            if self.squares[rank as usize][file as usize] != Piece::Empty {
                return false;
            }

            rank += rank_step;
            file += file_step;
        }

        true
    }
    fn find_king(&self, color: Color) -> Option<(usize, usize)> {
        let king = match color {
            Color::White => Piece::WhiteKing,
            Color::Black => Piece::BlackKing,
        };

        for rank in 0..8 {
            for file in 0..8 {
                if self.squares[rank][file] == king {
                    return Some((rank, file));
                }
            }
        }

        None
    }
    pub fn square_to_string(square: (usize, usize)) -> String {
        let (rank, file) = square;

        let file_char = (b'a' + file as u8) as char;
        let rank_char = (b'8' - rank as u8) as char;

        format!("{}{}", file_char, rank_char)
    }
    
    fn generate_pawn_moves(
        &self,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        let piece = self.squares[rank][file];

        let direction: isize;
        let start_rank: usize;
        let promotion_rank: usize;

        match piece {
            Piece::WhitePawn => {
                direction = -1;
                start_rank = 6;
                promotion_rank = 0;
            }

            Piece::BlackPawn => {
                direction = 1;
                start_rank = 1;
                promotion_rank = 7;
            }

            _ => return,
        }

        let next_rank = rank as isize + direction;

        if next_rank >= 0
            && next_rank < 8
            && self.squares[next_rank as usize][file] == Piece::Empty
        {
            let next_rank_usize = next_rank as usize;
            
            if next_rank_usize == promotion_rank {
                let promotion_pieces = match piece {
                    Piece::WhitePawn => [Piece::WhiteQueen, Piece::WhiteRook, Piece::WhiteBishop, Piece::WhiteKnight],
                    Piece::BlackPawn => [Piece::BlackQueen, Piece::BlackRook, Piece::BlackBishop, Piece::BlackKnight],
                    _ => unreachable!(),
                };
                
                for promo_piece in promotion_pieces {
                    moves.push(ChessMove::promotion(rank, file, next_rank_usize, file, promo_piece));
                }
            } else {
                moves.push(ChessMove::new(
                    rank,
                    file,
                    next_rank_usize,
                    file,
                ));

                if rank == start_rank {
                    let double_rank = rank as isize + direction * 2;

                    if self.squares[double_rank as usize][file] == Piece::Empty {
                        moves.push(ChessMove::new(
                            rank,
                            file,
                            double_rank as usize,
                            file,
                        ));
                    }
                }
            }
        }

        for file_offset in [-1isize, 1isize] {
            let target_file = file as isize + file_offset;

            if next_rank < 0
                || next_rank >= 8
                || target_file < 0
                || target_file >= 8
            {
                continue;
            }

            let next_rank_usize = next_rank as usize;
            let target_file_usize = target_file as usize;
            let target = self.squares[next_rank_usize][target_file_usize];

            if target != Piece::Empty
                && same_color(piece, target) == false
            {
                if next_rank_usize == promotion_rank {
                    let promotion_pieces = match piece {
                        Piece::WhitePawn => [Piece::WhiteQueen, Piece::WhiteRook, Piece::WhiteBishop, Piece::WhiteKnight],
                        Piece::BlackPawn => [Piece::BlackQueen, Piece::BlackRook, Piece::BlackBishop, Piece::BlackKnight],
                        _ => unreachable!(),
                    };
                    
                    for promo_piece in promotion_pieces {
                        moves.push(ChessMove::promotion(rank, file, next_rank_usize, target_file_usize, promo_piece));
                    }
                } else {
                    moves.push(ChessMove::new(
                        rank,
                        file,
                        next_rank_usize,
                        target_file_usize,
                    ));
                }
            }
        }

        for file_offset in [-1isize, 1isize] {
            let target_file = file as isize + file_offset;

            if target_file < 0 || target_file >= 8 {
                continue;
            }

            if let Some((ep_rank, ep_file)) = self.en_passant_square {
                let next_rank_usize = next_rank as usize;
                if next_rank >= 0 && next_rank < 8 && next_rank_usize == ep_rank && target_file as usize == ep_file {
                    moves.push(ChessMove::en_passant(
                        rank,
                        file,
                        next_rank_usize,
                        target_file as usize,
                    ));
                }
            }
        }
    }
    fn generate_knight_moves(
        &self,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        let piece = self.squares[rank][file];

        let offsets = [
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ];

        for (dr, df) in offsets {
            let target_rank = rank as isize + dr;
            let target_file = file as isize + df;

            if target_rank < 0
                || target_rank >= 8
                || target_file < 0
                || target_file >= 8
            {
                continue;
            }

            let target = self.squares[target_rank as usize][target_file as usize];

            if target == Piece::Empty || !same_color(piece, target) {
                moves.push(ChessMove::new(
                    rank,
                    file,
                    target_rank as usize,
                    target_file as usize,
                ));
            }
        }
    }
    fn generate_bishop_moves(
        &self,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        let piece = self.squares[rank][file];

        let directions = [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ];

        for (dr, df) in directions {
            let mut r = rank as isize + dr;
            let mut f = file as isize + df;

            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                let target = self.squares[r as usize][f as usize];

                if target == Piece::Empty {
                    moves.push(ChessMove::new(
                        rank,
                        file,
                        r as usize,
                        f as usize,
                    ));
                } else {
                    if !same_color(piece, target) {
                        moves.push(ChessMove::new(
                            rank,
                            file,
                            r as usize,
                            f as usize,
                        ));
                    }

                    break;
                }

                r += dr;
                f += df;
            }
        }
    }
    fn generate_rook_moves(
        &self,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        let piece = self.squares[rank][file];

        let directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ];

        for (dr, df) in directions {
            let mut r = rank as isize + dr;
            let mut f = file as isize + df;

            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                let target = self.squares[r as usize][f as usize];

                if target == Piece::Empty {
                    moves.push(ChessMove::new(
                        rank,
                        file,
                        r as usize,
                        f as usize,
                    ));
                } else {
                    if !same_color(piece, target) {
                        moves.push(ChessMove::new(
                            rank,
                            file,
                            r as usize,
                            f as usize,
                        ));
                    }

                    break;
                }

                r += dr;
                f += df;
            }
        }
    }
    fn generate_king_moves(
        &self,
        rank: usize,
        file: usize,
        moves: &mut Vec<ChessMove>,
    ) {
        let piece = self.squares[rank][file];

        for dr in -1isize..=1 {
            for df in -1isize..=1 {
                if dr == 0 && df == 0 {
                    continue;
                }

                let target_rank = rank as isize + dr;
                let target_file = file as isize + df;

                if target_rank < 0
                    || target_rank >= 8
                    || target_file < 0
                    || target_file >= 8
                {
                    continue;
                }

                let target = self.squares[target_rank as usize][target_file as usize];

                if target == Piece::Empty || !same_color(piece, target) {
                    moves.push(ChessMove::new(
                        rank,
                        file,
                        target_rank as usize,
                        target_file as usize,
                    ));
                }
            }
        }

        match piece {
            Piece::WhiteKing => {
                if self.castling_rights.white_kingside
                    && rank == 7
                    && file == 4
                    && self.squares[7][5] == Piece::Empty
                    && self.squares[7][6] == Piece::Empty
                    && self.squares[7][7] == Piece::WhiteRook
                    && !self.is_in_check(Color::White)
                    && !self.is_square_attacked((7, 4), Color::Black)
                    && !self.is_square_attacked((7, 5), Color::Black)
                {
                    moves.push(ChessMove::castling(rank, file, rank, 6));
                }

                if self.castling_rights.white_queenside
                    && rank == 7
                    && file == 4
                    && self.squares[7][3] == Piece::Empty
                    && self.squares[7][2] == Piece::Empty
                    && self.squares[7][1] == Piece::Empty
                    && self.squares[7][0] == Piece::WhiteRook
                    && !self.is_in_check(Color::White)
                    && !self.is_square_attacked((7, 4), Color::Black)
                    && !self.is_square_attacked((7, 3), Color::Black)
                {
                    moves.push(ChessMove::castling(rank, file, rank, 2));
                }
            }

            Piece::BlackKing => {
                if self.castling_rights.black_kingside
                    && rank == 0
                    && file == 4
                    && self.squares[0][5] == Piece::Empty
                    && self.squares[0][6] == Piece::Empty
                    && self.squares[0][7] == Piece::BlackRook
                    && !self.is_in_check(Color::Black)
                    && !self.is_square_attacked((0, 4), Color::White)
                    && !self.is_square_attacked((0, 5), Color::White)
                {
                    moves.push(ChessMove::castling(rank, file, rank, 6));
                }

                if self.castling_rights.black_queenside
                    && rank == 0
                    && file == 4
                    && self.squares[0][3] == Piece::Empty
                    && self.squares[0][2] == Piece::Empty
                    && self.squares[0][1] == Piece::Empty
                    && self.squares[0][0] == Piece::BlackRook
                    && !self.is_in_check(Color::Black)
                    && !self.is_square_attacked((0, 4), Color::White)
                    && !self.is_square_attacked((0, 3), Color::White)
                {
                    moves.push(ChessMove::castling(rank, file, rank, 2));
                }
            }

            _ => {}
        }
    }
}