use std::str::FromStr;

#[derive(clap::Parser, Debug)]
struct Args {
    #[arg(long, default_value = "config.yaml")]
    config: std::path::PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    List,
    View {
        #[arg(long)]
        task_id: usize,
        #[arg(long, default_value = "output.html")]
        output: std::path::PathBuf,
    },
}

#[derive(Debug, serde::Serialize)]
struct MeshData {
    vertices: Vec<f32>,
    indices: Vec<u32>,
    classes: Vec<Vec<f32>>,
}

#[tokio::main]
async fn main() {
    let args = <Args as clap::Parser>::parse();

    let mut file = std::fs::File::open(&args.config)
        .expect("failed to open config file");
    let reader = std::io::BufReader::new(&mut file);
    let config: serde_yaml::Value = serde_yaml::from_reader(reader).unwrap();
    let database_url = config
        .get("global")
        .and_then(|v| v.get("database"))
        .and_then(|v| v.as_str())
        .expect("database_path not found in config");
    let opt = sqlx::sqlite::SqliteConnectOptions::from_str(database_url)
        .unwrap()
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(opt).await.unwrap();

    match args.command {
        Command::List => list_tasks(&pool).await,
        Command::View { task_id, output } => view_task(&pool, task_id, output).await,
    }
}

async fn list_tasks(pool: &sqlx::SqlitePool) {
    let task_rows: Vec<(i64, String, String)> = sqlx::query_as("SELECT id, task_name, params_json FROM tasks WHERE task_name = 'FinalizeTask' ORDER BY id")
        .fetch_all(pool)
        .await
        .unwrap();
    println!("tasks:");
    for (id, task_name, params_json) in task_rows.iter() {
        println!("\tid: {}, name: {}, params: {}", id, task_name, params_json);
    }
}

async fn view_task(pool: &sqlx::SqlitePool, task_id: usize, output: std::path::PathBuf) {
    let artifact: (Vec<u8>,) = sqlx::query_as("SELECT artifact FROM tasks WHERE id = ?")
        .bind(task_id as i64)
        .fetch_one(pool)
        .await
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let decoder = flate2::read::GzDecoder::new(artifact.0.as_slice());
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(dir.path()).unwrap();
    println!("extracted artifact.");
    
    let database_url = dir.path().join("finalize.db")
        .to_string_lossy().to_string();
    let opt = sqlx::sqlite::SqliteConnectOptions::from_str(&database_url)
        .unwrap()
        .create_if_missing(true);
    let pool = sqlx::SqlitePool::connect_with(opt).await.unwrap();

    let vertex_rows: Vec<(f64, f64, f64)> = sqlx::query_as("SELECT x, y, z FROM verts ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();
    let index_rows: Vec<(i64, i64, i64)> = sqlx::query_as("SELECT v1, v2, v3 FROM faces ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();
    let class_rows: Vec<(String,)> = sqlx::query_as("SELECT class_vec FROM class_vecs ORDER BY id")
        .fetch_all(&pool)
        .await
        .unwrap();
    println!("vertices: {}, indices: {}, classes: {}", vertex_rows.len(), index_rows.len(), class_rows.len());

    let mut vertices = Vec::with_capacity(vertex_rows.len() * 3);
    for (x, y, z) in vertex_rows.iter() {
        vertices.push(*x as f32);
        vertices.push(*y as f32);
        vertices.push(*z as f32);
    }
    let mut indices = Vec::with_capacity(index_rows.len() * 3);
    for (v1, v2, v3) in index_rows.iter() {
        indices.push(*v1 as u32);
        indices.push(*v2 as u32);
        indices.push(*v3 as u32);
    }
    let mut classes = Vec::with_capacity(class_rows.len());
    for (class_vec,) in class_rows.iter() {
        let vec: Vec<f32> = serde_json::from_str(class_vec).unwrap();
        classes.push(vec);
    }
    let n_class = classes[0].len();
    let mut vclasses = vec![vec![0.0; n_class]; vertex_rows.len()];
    for (i, class_vec) in classes.iter().enumerate() {
        let indices_slice = &indices[i * 3..i * 3 + 3];
        for j in 0..n_class {
            vclasses[indices_slice[0] as usize][j] += class_vec[j];
            vclasses[indices_slice[1] as usize][j] += class_vec[j];
            vclasses[indices_slice[2] as usize][j] += class_vec[j];
        }
    }
    let mesh_data = MeshData { vertices, indices, classes: vclasses };

    // export html
    let mut jinja_env = minijinja::Environment::new();
    jinja_env.add_template("index.html", include_str!("index.html")).unwrap();
    let template = jinja_env.get_template("index.html").unwrap();
    let html = template.render(minijinja::context! { mesh_data }).unwrap();
    std::fs::write(&output, html).unwrap();
    println!("generated html output: {}", output.display());
}
