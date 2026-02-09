use rusqlite::{params, Connection, Result};

use crate::reduce::ReductionStatus;

pub struct Db {
    pub conn: Connection,
}

impl Db {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        let db = Db { conn };
        db.create_tables()?;
        Ok(db)
    }

    fn create_tables(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                basis TEXT NOT NULL,
                fuel INTEGER NOT NULL,
                max_term_size INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(name, basis, fuel, max_term_size)
            );

            CREATE TABLE IF NOT EXISTS reductions (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                size INTEGER NOT NULL,
                term TEXT NOT NULL,
                normal_form TEXT,
                steps INTEGER NOT NULL,
                status TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_reductions_exp_size
                ON reductions(experiment_id, size);
            CREATE INDEX IF NOT EXISTS idx_reductions_nf
                ON reductions(normal_form) WHERE normal_form IS NOT NULL;

            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                size INTEGER NOT NULL,
                total_terms INTEGER NOT NULL,
                distinct_nfs INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                divergent_count INTEGER NOT NULL,
                explosive_count INTEGER NOT NULL,
                motif_count INTEGER NOT NULL,
                avg_steps REAL NOT NULL,
                max_steps INTEGER NOT NULL,
                UNIQUE(experiment_id, size)
            );

            CREATE TABLE IF NOT EXISTS motifs (
                id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL REFERENCES experiments(id),
                size INTEGER NOT NULL,
                term TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                nf_count INTEGER NOT NULL,
                savings INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_motifs_exp_size
                ON motifs(experiment_id, size);
            ",
        )
    }

    /// Get or create an experiment, returns its id.
    pub fn get_or_create_experiment(
        &self,
        name: &str,
        basis: &str,
        fuel: usize,
        max_term_size: usize,
    ) -> Result<i64> {
        // Try to find existing
        let mut stmt = self.conn.prepare(
            "SELECT id FROM experiments WHERE name=?1 AND basis=?2 AND fuel=?3 AND max_term_size=?4",
        )?;
        let mut rows = stmt.query(params![name, basis, fuel as i64, max_term_size as i64])?;
        if let Some(row) = rows.next()? {
            return row.get(0);
        }
        drop(rows);
        drop(stmt);

        self.conn.execute(
            "INSERT INTO experiments (name, basis, fuel, max_term_size) VALUES (?1, ?2, ?3, ?4)",
            params![name, basis, fuel as i64, max_term_size as i64],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Check if a size has already been fully computed for an experiment.
    pub fn size_complete(&self, experiment_id: i64, size: usize) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM reports WHERE experiment_id=?1 AND size=?2",
            params![experiment_id, size as i64],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Store a batch of reduction results. Uses a transaction for speed.
    pub fn store_reductions(
        &self,
        experiment_id: i64,
        size: usize,
        results: &[(String, Option<String>, usize, ReductionStatus)],
    ) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO reductions (experiment_id, size, term, normal_form, steps, status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (term, nf, steps, status) in results {
                let status_str = match status {
                    ReductionStatus::NormalForm => "normal",
                    ReductionStatus::OutOfFuel => "divergent",
                    ReductionStatus::TermTooLarge => "explosive",
                };
                stmt.execute(params![
                    experiment_id,
                    size as i64,
                    term,
                    nf.as_deref(),
                    *steps as i64,
                    status_str,
                ])?;
            }
        }
        tx.commit()
    }

    /// Store a size report.
    pub fn store_report(
        &self,
        experiment_id: i64,
        size: usize,
        total_terms: usize,
        distinct_nfs: usize,
        compression_ratio: f64,
        divergent_count: usize,
        explosive_count: usize,
        motif_count: usize,
        avg_steps: f64,
        max_steps: usize,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO reports
             (experiment_id, size, total_terms, distinct_nfs, compression_ratio,
              divergent_count, explosive_count, motif_count, avg_steps, max_steps)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                experiment_id,
                size as i64,
                total_terms as i64,
                distinct_nfs as i64,
                compression_ratio,
                divergent_count as i64,
                explosive_count as i64,
                motif_count as i64,
                avg_steps,
                max_steps as i64,
            ],
        )?;
        Ok(())
    }

    /// Store reuse/motif candidates.
    pub fn store_motifs(
        &self,
        experiment_id: i64,
        size: usize,
        motifs: &[(String, usize, usize)],
    ) -> Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        // Clear old motifs for this experiment+size
        tx.execute(
            "DELETE FROM motifs WHERE experiment_id=?1 AND size=?2",
            params![experiment_id, size as i64],
        )?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO motifs (experiment_id, size, term, frequency, nf_count, savings)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (term, nf_count, savings) in motifs {
                stmt.execute(params![
                    experiment_id,
                    size as i64,
                    term,
                    0i64, // frequency (we store nf_count in frequency for now)
                    *nf_count as i64,
                    *savings as i64,
                ])?;
            }
        }
        tx.commit()
    }

    /// Load all reports for an experiment, ordered by size.
    pub fn load_reports(&self, experiment_id: i64) -> Result<Vec<ReportRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT size, total_terms, distinct_nfs, compression_ratio,
                    divergent_count, explosive_count, motif_count, avg_steps, max_steps
             FROM reports WHERE experiment_id=?1 ORDER BY size",
        )?;
        let rows = stmt.query_map(params![experiment_id], |row| {
            Ok(ReportRow {
                size: row.get::<_, i64>(0)? as usize,
                total_terms: row.get::<_, i64>(1)? as usize,
                distinct_nfs: row.get::<_, i64>(2)? as usize,
                compression_ratio: row.get(3)?,
                divergent_count: row.get::<_, i64>(4)? as usize,
                explosive_count: row.get::<_, i64>(5)? as usize,
                motif_count: row.get::<_, i64>(6)? as usize,
                avg_steps: row.get(7)?,
                max_steps: row.get::<_, i64>(8)? as usize,
            })
        })?;
        rows.collect()
    }
}

#[derive(Debug)]
pub struct ReportRow {
    pub size: usize,
    pub total_terms: usize,
    pub distinct_nfs: usize,
    pub compression_ratio: f64,
    pub divergent_count: usize,
    pub explosive_count: usize,
    pub motif_count: usize,
    pub avg_steps: f64,
    pub max_steps: usize,
}
