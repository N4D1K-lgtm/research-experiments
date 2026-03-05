mod schema;
mod store;

use std::path::PathBuf;

use anyhow::Result;
use async_graphql::http::GraphiQLSource;
use async_graphql_axum::{GraphQLRequest, GraphQLResponse};
use axum::{
    Router,
    extract::State,
    response::{Html, IntoResponse},
    routing::get,
};
use clap::Parser;
use tower_http::cors::{Any, CorsLayer};

use ipa_core::Language;
use ipa_db::Database;
use ipa_trie::{ConeTreeLayout, PhonologicalTrie};

use schema::{IpaSchema, build_schema};
use store::TrieStore;

#[derive(Parser)]
#[command(name = "ipa-server")]
#[command(about = "GraphQL API server for IPA trie visualization")]
struct Cli {
    /// Database path
    #[arg(long, default_value = "data/ipa.db")]
    db_path: PathBuf,

    /// Server port
    #[arg(long, default_value = "3001")]
    port: u16,

    /// Minimum count threshold for pruning
    #[arg(long, default_value = "2")]
    min_count: u64,

    /// Language codes (comma-separated, or "all")
    #[arg(long, default_value = "all")]
    lang: String,
}

async fn graphql_handler(
    State(schema): State<IpaSchema>,
    req: GraphQLRequest,
) -> GraphQLResponse {
    schema.execute(req.into_inner()).await.into()
}

async fn graphiql_handler() -> impl IntoResponse {
    Html(
        GraphiQLSource::build()
            .endpoint("/graphql")
            .finish(),
    )
}

fn resolve_languages(lang_arg: &str) -> Vec<Language> {
    if lang_arg == "all" {
        Language::all()
    } else {
        lang_arg
            .split(',')
            .filter_map(|code| Language::by_code(code.trim()))
            .collect()
    }
}

async fn build_trie(db: &Database, languages: &[Language], min_count: u64) -> Result<PhonologicalTrie> {
    let mut trie = PhonologicalTrie::new();

    for lang in languages {
        let pronunciations = db.get_pronunciations_for_language(&lang.code).await?;
        tracing::info!("{}: {} pronunciations", lang.code, pronunciations.len());
        for (word, phonemes) in &pronunciations {
            trie.insert(phonemes, &lang.code, word);
        }
    }

    if min_count > 1 {
        trie.prune(min_count);
    }

    trie.assign_ids();
    trie.classify_positions();
    trie.compute_transition_probs();
    trie.compute_weights();
    trie.assign_colors();
    ConeTreeLayout::apply(&mut trie.root);

    tracing::info!(
        "Built trie: {} nodes, max_depth={}, terminals={}",
        trie.node_count,
        trie.max_depth(),
        trie.terminal_count()
    );

    Ok(trie)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let cli = Cli::parse();
    let languages = resolve_languages(&cli.lang);

    tracing::info!("Opening database: {}", cli.db_path.display());
    let db = Database::open(&cli.db_path).await?;
    db.init_schema().await?;

    tracing::info!("Building trie from {} languages...", languages.len());
    let trie = build_trie(&db, &languages, cli.min_count).await?;

    tracing::info!("Flattening trie to in-memory store...");
    let store = TrieStore::from_trie(&trie);
    tracing::info!(
        "Store ready: {} nodes, {} edges",
        store.metadata.node_count,
        store.metadata.edge_count
    );

    let schema = build_schema(store);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/graphql", get(graphiql_handler).post(graphql_handler))
        .layer(cors)
        .with_state(schema);

    let addr = format!("0.0.0.0:{}", cli.port);
    tracing::info!("Server listening on http://localhost:{}", cli.port);
    tracing::info!("GraphQL Playground: http://localhost:{}/graphql", cli.port);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
