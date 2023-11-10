use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// I checked that names.txt doesn't have either of these characters, so they're safe to use as start and stop tokens.
// This is obviously not a fully general solution (also, it only works when tokens are characters)
static START_TOKEN: char = '<';
static END_TOKEN: char = '>';

fn lines_from_file<P>(filepath: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filepath)?;
    Ok(io::BufReader::new(file).lines())
}

fn print_entry(entry: &(&(char, char), &i32)) {
    let (k, v) = entry;
    println!("({},{}): {}", k.0, k.1,v);
}

pub fn main() {
    env_logger::init();

    let path = Path::new("data/names.txt");
    let words: Vec<String> = lines_from_file(path).expect("").map(|l| l.unwrap()).collect();
    log::debug!("Read {} names from {}", words.len(), path.display());

    if log::log_enabled!(log::Level::Info) {
        println!("\n{}\n", words[0..10].join("\n"));

        let min = words.iter().map(|l| l.len()).min().unwrap();
        println!("Shortest name length: {}", min);
    
        let max = words.iter().map(|l| l.len()).max().unwrap();
        println!("Longest name length: {}\n", max);
        
        let mut word = words[0].clone();
        word.insert_str(0, &START_TOKEN.to_string());
        word.push(END_TOKEN);
        let chars = word.chars();
        println!("Bigrams:");
        for (c1, c2) in chars.clone().zip(chars.skip(1)) {
            println!("\t{} {}", c1, c2);
        }
        println!("");
    }

    let mut bigrams = HashMap::<(char, char), i32>::new();
    for mut word in words.into_iter() {
        word.insert_str(0, &START_TOKEN.to_string());
        word.push(END_TOKEN);
        let chars = word.chars();
        for (c1, c2) in chars.clone().zip(chars.skip(1)) {
            *bigrams.entry((c1, c2)).or_insert(0) += 1;
        }
    }
    if log::log_enabled!(log::Level::Info) {
        let mut entries: Vec<(_,_)> = bigrams.iter().collect();
        entries.sort_by(|a,b| a.1.cmp(&b.1));
        entries.reverse();
        entries.iter().take(10).for_each(print_entry);
    }

}