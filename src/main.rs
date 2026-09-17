mod board;

use board::Board;

fn main() {
    let board = Board::new();

    board.print();

    let moves = board.generate_moves();

    println!("Legal moves: {}", moves.len());

    for mv in &moves {
        println!(
            "{}{}",
            Board::square_to_string_public(mv.from),
            Board::square_to_string_public(mv.to)
        );
    }
}