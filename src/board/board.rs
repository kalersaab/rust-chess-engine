use super::pieces::*;
use super::chess_move::MoveType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

impl CastlingRights {
    pub fn new() -> Self {
        CastlingRights {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        }
    }

    pub fn to_string(&self) -> String {
        let mut result = String::new();
        if self.white_kingside {
            result.push('K');
        }
        if self.white_queenside {
            result.push('Q');
        }
        if self.black_kingside {
            result.push('k');
        }
        if self.black_queenside {
            result.push('q');
        }
        if result.is_empty() {
            result.push('-');
        }
        result
    }

    pub fn from_string(s: &str) -> Result<Self, String> {
        if s == "-" {
            return Ok(CastlingRights {
                white_kingside: false,
                white_queenside: false,
                black_kingside: false,
                black_queenside: false,
            });
        }

        let mut rights = CastlingRights {
            white_kingside: false,
            white_queenside: false,
            black_kingside: false,
            black_queenside: false,
        };

        for c in s.chars() {
            match c {
                'K' => rights.white_kingside = true,
                'Q' => rights.white_queenside = true,
                'k' => rights.black_kingside = true,
                'q' => rights.black_queenside = true,
                _ => return Err(format!("Invalid castling right: {}", c)),
            }
        }

        Ok(rights)
    }
}

#[derive(Clone)]
pub struct Board {
    pub squares: [[Piece; 8]; 8],
    pub turn: Color,
    pub castling_rights: CastlingRights,
    pub en_passant_square: Option<(usize, usize)>,
    pub halfmove_clock: u32,
    pub fullmove_number: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameStatus {
    Playing,
    Check,
    Checkmate,
    Stalemate,
}

impl Board {
    pub fn square_to_string_public(
        square: (usize, usize),
    ) -> String {
        Self::square_to_string(square)
    }

    pub fn parse_square_public(square: &str) -> Result<(usize, usize), String> {
        parse_square(square)
    }
    pub fn new() -> Self {
        let mut board = Board {
            squares: [[Piece::Empty; 8]; 8],
            turn: Color::White,
            castling_rights: CastlingRights::new(),
            en_passant_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
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

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() != 6 {
            return Err("FEN must have 6 space-separated fields".to_string());
        }

        let mut board = Board {
            squares: [[Piece::Empty; 8]; 8],
            turn: Color::White,
            castling_rights: CastlingRights::new(),
            en_passant_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        };
        
        let ranks: Vec<&str> = parts[0].split('/').collect();
        if ranks.len() != 8 {
            return Err("Invalid piece placement".to_string());
        }

        for (rank_idx, rank_str) in ranks.iter().enumerate() {
            let mut file_idx = 0;
            for c in rank_str.chars() {
                if file_idx >= 8 {
                    return Err("Too many files in rank".to_string());
                }

                if c.is_ascii_digit() {
                    file_idx += c.to_digit(10).unwrap() as usize;
                } else {
                    board.squares[rank_idx][file_idx] =
                        char_to_piece(c)?;
                    file_idx += 1;
                }
            }
            if file_idx != 8 {
                return Err("Invalid number of files in rank".to_string());
            }
        }

        board.turn = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid turn".to_string()),
        };

        board.castling_rights = CastlingRights::from_string(parts[2])?;

        board.en_passant_square = match parts[3] {
            "-" => None,
            sq => Some(parse_square(sq)?),
        };

        board.halfmove_clock =
            parts[4].parse().map_err(|_| "Invalid halfmove clock".to_string())?;

        board.fullmove_number =
            parts[5].parse().map_err(|_| "Invalid fullmove number".to_string())?;

        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in 0..8 {
            let mut empty_count = 0;
            for file in 0..8 {
                let piece = self.squares[rank][file];
                if piece == Piece::Empty {
                    empty_count += 1;
                } else {
                    if empty_count > 0 {
                        fen.push_str(&empty_count.to_string());
                        empty_count = 0;
                    }
                    fen.push(piece_to_fen_char(piece));
                }
            }
            if empty_count > 0 {
                fen.push_str(&empty_count.to_string());
            }
            if rank < 7 {
                fen.push('/');
            }
        }

        fen.push(' ');

        fen.push(match self.turn {
            Color::White => 'w',
            Color::Black => 'b',
        });

        fen.push(' ');

        fen.push_str(&self.castling_rights.to_string());

        fen.push(' ');

        match self.en_passant_square {
            Some(sq) => fen.push_str(&Self::square_to_string(sq)),
            None => fen.push('-'),
        }

        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());

        fen.push(' ');

        fen.push_str(&self.fullmove_number.to_string());

        fen
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
            return Err(format!("It is {:?}'s turn", self.turn));
        }

        if !self.is_valid_piece_move(piece, from_rank, from_file, to_rank, to_file) {
            return Err(format!("Illegal {} -> {}", from, to));
        }

        let destination = self.squares[to_rank][to_file];

        if destination != Piece::Empty && same_color(piece, destination) {
            return Err("Cannot capture your own piece".to_string());
        }

        let mut move_type = MoveType::Normal;
        
        if piece == Piece::WhiteKing || piece == Piece::BlackKing {
            if (from_file as isize - to_file as isize).abs() == 2 {
                move_type = MoveType::Castling;
            }
        } else if piece == Piece::WhitePawn || piece == Piece::BlackPawn {
            if let Some((ep_rank, ep_file)) = self.en_passant_square {
                if to_rank == ep_rank && to_file == ep_file && destination == Piece::Empty {
                    move_type = MoveType::EnPassant;
                }
            }
            if to_rank == 0 || to_rank == 7 {
                let promo_piece = match piece {
                    Piece::WhitePawn => Piece::WhiteQueen,
                    Piece::BlackPawn => Piece::BlackQueen,
                    _ => unreachable!(),
                };
                move_type = MoveType::Promotion(promo_piece);
            }
        }

        self.execute_move((from_rank, from_file), (to_rank, to_file), move_type)?;

        Ok(())
    }

    pub fn execute_move(&mut self, from: (usize, usize), to: (usize, usize), move_type: MoveType) -> Result<(), String> {
        let (from_rank, from_file) = from;
        let (to_rank, to_file) = to;

        let piece = self.squares[from_rank][from_file];

        if piece == Piece::Empty {
            return Err("Cannot move from empty square".to_string());
        }

        self.en_passant_square = None;

        let destination = self.squares[to_rank][to_file];
        if piece == Piece::WhitePawn || piece == Piece::BlackPawn
            || destination != Piece::Empty
            || matches!(move_type, MoveType::EnPassant | MoveType::Promotion(_))
        {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        if piece == Piece::WhitePawn && from_rank == 6 && to_rank == 4 {
            self.en_passant_square = Some((5, from_file));
        } else if piece == Piece::BlackPawn && from_rank == 1 && to_rank == 3 {
            self.en_passant_square = Some((2, from_file));
        }

        match piece {
            Piece::WhiteKing => {
                self.castling_rights.white_kingside = false;
                self.castling_rights.white_queenside = false;
            }
            Piece::BlackKing => {
                self.castling_rights.black_kingside = false;
                self.castling_rights.black_queenside = false;
            }
            _ => {}
        }

        if piece == Piece::WhiteRook {
            if from_rank == 7 && from_file == 0 {
                self.castling_rights.white_queenside = false;
            } else if from_rank == 7 && from_file == 7 {
                self.castling_rights.white_kingside = false;
            }
        } else if piece == Piece::BlackRook {
            if from_rank == 0 && from_file == 0 {
                self.castling_rights.black_queenside = false;
            } else if from_rank == 0 && from_file == 7 {
                self.castling_rights.black_kingside = false;
            }
        }

        if destination == Piece::WhiteRook {
            if to_rank == 7 && to_file == 0 {
                self.castling_rights.white_queenside = false;
            } else if to_rank == 7 && to_file == 7 {
                self.castling_rights.white_kingside = false;
            }
        } else if destination == Piece::BlackRook {
            if to_rank == 0 && to_file == 0 {
                self.castling_rights.black_queenside = false;
            } else if to_rank == 0 && to_file == 7 {
                self.castling_rights.black_kingside = false;
            }
        }

        match move_type {
            MoveType::Castling => {
                self.squares[to_rank][to_file] = piece;
                self.squares[from_rank][from_file] = Piece::Empty;

                match self.turn {
                    Color::White => {
                        if to_file == 6 {
                            self.squares[7][5] = self.squares[7][7];
                            self.squares[7][7] = Piece::Empty;
                        } else if to_file == 2 {
                            self.squares[7][3] = self.squares[7][0];
                            self.squares[7][0] = Piece::Empty;
                        }
                    }
                    Color::Black => {
                        if to_file == 6 {
                            self.squares[0][5] = self.squares[0][7];
                            self.squares[0][7] = Piece::Empty;
                        } else if to_file == 2 {
                            self.squares[0][3] = self.squares[0][0];
                            self.squares[0][0] = Piece::Empty;
                        }
                    }
                }
            }
            MoveType::EnPassant => {
                self.squares[to_rank][to_file] = piece;
                self.squares[from_rank][from_file] = Piece::Empty;

                match piece {
                    Piece::WhitePawn => {
                        self.squares[to_rank + 1][to_file] = Piece::Empty;
                    }
                    Piece::BlackPawn => {
                        self.squares[to_rank - 1][to_file] = Piece::Empty;
                    }
                    _ => {}
                }
            }
            MoveType::Promotion(promotion_piece) => {
                self.squares[to_rank][to_file] = promotion_piece;
                self.squares[from_rank][from_file] = Piece::Empty;
            }
            MoveType::Normal => {
                self.squares[to_rank][to_file] = piece;
                self.squares[from_rank][from_file] = Piece::Empty;
            }
        }

        self.turn = match self.turn {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        if self.turn == Color::White {
            self.fullmove_number += 1;
        }

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

                (rank_diff == 0 && file_diff == 2) || (rank_diff <= 1 && file_diff <= 1)
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
        
        if file_diff == 0
            && rank_diff == direction
            && destination == Piece::Empty
        {
            return true;
        }

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