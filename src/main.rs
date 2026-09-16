mod board;

use board::Board;

fn main() {
    let mut board = Board::new();

    board.print();

    println!("White: e2e4");

    match board.make_move("e2", "e4") {
        Ok(_) => board.print(),
        Err(error) => println!("Error: {}", error),
    }

    println!("Black: e7e5");

    match board.make_move("e7", "e5") {
        Ok(_) => board.print(),
        Err(error) => println!("Error: {}", error),
    }

    println!("White: g1f3");

    match board.make_move("g1", "f3") {
        Ok(_) => board.print(),
        Err(error) => println!("Error: {}", error),
    }

    println!("Black: b8c6");

    match board.make_move("b8", "c6") {
        Ok(_) => board.print(),
        Err(error) => println!("Error: {}", error),
    }
}