use super::pieces::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone)]
pub struct Board {
    pub squares: [[Piece; 8]; 8],
    pub turn: Color,
}

impl Board {
    pub fn square_to_string_public(
        square: (usize, usize),
    ) -> String {
        Self::square_to_string(square)
    }
    pub fn new() -> Self {
        let mut board = Board {
            squares: [[Piece::Empty; 8]; 8],
            turn: Color::White,
        };

        board.squares[0] = [
            Piece::BlackRook,
            Piece::BlackKnight,
            Piece::BlackBishop,
            Piece::BlackQueen,
            Piece::BlackKing,
            Piece::BlackBishop,
            Piece::BlackKnight,
            Piece::BlackRook,
        ];

        board.squares[1] = [Piece::BlackPawn; 8];

        board.squares[6] = [Piece::WhitePawn; 8];

        board.squares[7] = [
            Piece::WhiteRook,
            Piece::WhiteKnight,
            Piece::WhiteBishop,
            Piece::WhiteQueen,
            Piece::WhiteKing,
            Piece::WhiteBishop,
            Piece::WhiteKnight,
            Piece::WhiteRook,
        ];

        board
    }

    pub fn print(&self) {
        println!();
        println!("  a b c d e f g h");

        for rank in 0..8 {
            print!("{} ", 8 - rank);

            for file in 0..8 {
                let piece = self.squares[rank][file];
                print!("{} ", piece_to_char(piece));
            }

            println!("{}", 8 - rank);
        }

        println!("  a b c d e f g h");

        match self.turn {
            Color::White => println!("Turn: White"),
            Color::Black => println!("Turn: Black"),
        }

        println!();
    }

    pub fn make_move(&mut self, from: &str, to: &str) -> Result<(), String> {
        let (from_rank, from_file) = parse_square(from)?;
        let (to_rank, to_file) = parse_square(to)?;

        let piece = self.squares[from_rank][from_file];

        if piece == Piece::Empty {
            return Err(format!("No piece on {}", from));
        }

        if !self.is_correct_turn(piece) {
            return Err(format!(
                "It is {:?}'s turn",
                self.turn
            ));
        }

        if !self.is_valid_piece_move(
            piece,
            from_rank,
            from_file,
            to_rank,
            to_file,
        ) {
            return Err(format!(
                "Illegal {} -> {}",
                from, to
            ));
        }

        let destination = self.squares[to_rank][to_file];

        // Cannot capture your own piece
        if destination != Piece::Empty
            && same_color(piece, destination)
        {
            return Err("Cannot capture your own piece".to_string());
        }

        // Move piece
        self.squares[to_rank][to_file] = piece;
        self.squares[from_rank][from_file] = Piece::Empty;

        // Change turn
        self.turn = match self.turn {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        Ok(())
    }

    fn is_correct_turn(&self, piece: Piece) -> bool {
        match self.turn {
            Color::White => is_white(piece),
            Color::Black => is_black(piece),
        }
    }

    fn is_valid_piece_move(
        &self,
        piece: Piece,
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
    ) -> bool {
        match piece {
            Piece::WhitePawn => {
                self.valid_pawn_move(
                    from_rank,
                    from_file,
                    to_rank,
                    to_file,
                    Color::White,
                )
            }

            Piece::BlackPawn => {
                self.valid_pawn_move(
                    from_rank,
                    from_file,
                    to_rank,
                    to_file,
                    Color::Black,
                )
            }

            Piece::WhiteKnight | Piece::BlackKnight => {
                let rank_diff =
                    (to_rank as i32 - from_rank as i32).abs();

                let file_diff =
                    (to_file as i32 - from_file as i32).abs();

                (rank_diff == 2 && file_diff == 1)
                    || (rank_diff == 1 && file_diff == 2)
            }

            Piece::WhiteKing | Piece::BlackKing => {
                let rank_diff =
                    (to_rank as i32 - from_rank as i32).abs();

                let file_diff =
                    (to_file as i32 - from_file as i32).abs();

                rank_diff <= 1 && file_diff <= 1
            }

            Piece::WhiteBishop | Piece::BlackBishop => {
                self.valid_sliding_move(
                    from_rank,
                    from_file,
                    to_rank,
                    to_file,
                    true,
                    false,
                )
            }

            Piece::WhiteRook | Piece::BlackRook => {
                self.valid_sliding_move(
                    from_rank,
                    from_file,
                    to_rank,
                    to_file,
                    false,
                    true,
                )
            }

            Piece::WhiteQueen | Piece::BlackQueen => {
                self.valid_sliding_move(
                    from_rank,
                    from_file,
                    to_rank,
                    to_file,
                    true,
                    true,
                )
            }

            Piece::Empty => false,
        }
    }

    fn valid_pawn_move(
        &self,
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
        color: Color,
    ) -> bool {
        let direction: i32 = match color {
            Color::White => -1,
            Color::Black => 1,
        };

        let start_rank = match color {
            Color::White => 6,
            Color::Black => 1,
        };

        let rank_diff =
            to_rank as i32 - from_rank as i32;

        let file_diff =
            to_file as i32 - from_file as i32;

        let destination = self.squares[to_rank][to_file];

        // One square forward
        if file_diff == 0
            && rank_diff == direction
            && destination == Piece::Empty
        {
            return true;
        }

        // Two squares from starting position
        if file_diff == 0
            && from_rank == start_rank
            && rank_diff == direction * 2
        {
            let middle_rank =
                (from_rank as i32 + direction) as usize;

            return self.squares[middle_rank][from_file]
                == Piece::Empty
                && destination == Piece::Empty;
        }

        // Capture diagonally
        if file_diff.abs() == 1
            && rank_diff == direction
            && destination != Piece::Empty
            && !same_color(
            self.squares[from_rank][from_file],
            destination,
        )
        {
            return true;
        }

        false
    }

    fn valid_sliding_move(
        &self,
        from_rank: usize,
        from_file: usize,
        to_rank: usize,
        to_file: usize,
        diagonal: bool,
        straight: bool,
    ) -> bool {
        let rank_diff =
            to_rank as i32 - from_rank as i32;

        let file_diff =
            to_file as i32 - from_file as i32;

        let is_diagonal =
            rank_diff.abs() == file_diff.abs();

        let is_straight =
            rank_diff == 0 || file_diff == 0;

        if diagonal && !straight && !is_diagonal {
            return false;
        }

        if straight && !diagonal && !is_straight {
            return false;
        }

        if diagonal && straight
            && !is_diagonal
            && !is_straight
        {
            return false;
        }

        let rank_step = rank_diff.signum();
        let file_step = file_diff.signum();

        let mut rank = from_rank as i32 + rank_step;
        let mut file = from_file as i32 + file_step;

        while rank != to_rank as i32
            || file != to_file as i32
        {
            if self.squares[rank as usize][file as usize]
                != Piece::Empty
            {
                return false;
            }

            rank += rank_step;
            file += file_step;
        }

        true
    }
}

fn parse_square(square: &str) -> Result<(usize, usize), String> {
    if square.len() != 2 {
        return Err(format!("Invalid square: {}", square));
    }

    let bytes = square.as_bytes();

    let file = match bytes[0] as char {
        'a'..='h' => (bytes[0] - b'a') as usize,
        _ => return Err(format!("Invalid file: {}", square)),
    };

    let rank = match bytes[1] as char {
        '1'..='8' => 8 - (bytes[1] - b'0') as usize,
        _ => return Err(format!("Invalid rank: {}", square)),
    };

    Ok((rank, file))
}