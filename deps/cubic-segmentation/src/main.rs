mod raycast;
mod triangle;

#[derive(clap::Parser, Debug)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Compute 3d object-space based on ray-casting from 2d segmentation-map.
    #[command(arg_required_else_help = true)]
    Raycast {
        /// Database file path (e.g. /path/to/database.db).
        database_path: String,
        /// Extract classes prompt from 2d segmentaion-map (e.g. "road,line,pole").
        prompt: String,
    },
}

fn main() {
    let args = <Args as clap::Parser>::parse();

    match &args.command {
        Commands::Raycast { database_path, prompt } => {
            let mut conn = rusqlite::Connection::open(database_path).unwrap();
            let class_names = prompt
                .split('.')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>();
            raycast::task_raycast(&mut conn, class_names.len(), &class_names);
        }
    }
}
