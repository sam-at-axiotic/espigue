//! KB manifest for incremental reindexing.
//!
//! Phase 3 Task 3.2. Tracks content hashes per KB file in `kb/INDEX.toml`
//! so `KbIndexer` (Task 3.3) only re-embeds files whose contents changed.
//! On first run, the file is created from scratch.
//!
//! AC-1: load failures (parse errors, version-too-new) surface as
//! `AlzinaError::Search` with `degraded=true` so the indexer can route the
//! degradation upward rather than silently swallowing the manifest.
//!
//! Manifest paths are stored *relative to `kb_root`* and always use
//! forward slashes so the file is portable across platforms.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use chrono::Utc;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use alzina_core::{AlzinaError, AlzinaResult, SearchDetail};

/// Current manifest format version. Bump when the on-disk shape changes.
pub const MANIFEST_VERSION: u32 = 1;

/// File name of the manifest within `kb_root`.
pub const MANIFEST_FILE: &str = "INDEX.toml";

/// Prefix shared by every per-save tempfile. Saves write to
/// `<prefix><uuid>.tmp` so two concurrent writers can't collide on the
/// same path. Documented as a constant so tooling and tests can spot
/// orphans without hard-coding the literal in multiple places.
const MANIFEST_TMP_PREFIX: &str = ".INDEX.toml.";

/// Suffix for per-save tempfiles. Combined with `MANIFEST_TMP_PREFIX`
/// and a UUID to form a unique path per save attempt.
const MANIFEST_TMP_SUFFIX: &str = ".tmp";

/// Sentinel file name for the advisory exclusive lock that ensures only
/// one process at a time mutates the manifest. Held for the lifetime of
/// the `KbManifest` instance and released on drop.
const MANIFEST_LOCK_FILE: &str = ".INDEX.lock";

/// Per-file metadata recorded after a successful index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    /// Content SHA-256 (hex) of the indexed file.
    pub content_hash: String,
    /// ISO-8601 timestamp of the indexing.
    pub last_indexed: String,
    /// Number of chunks produced from this file.
    pub chunk_count: usize,
}

/// On-disk shape of `kb/INDEX.toml`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ManifestData {
    #[serde(default)]
    pub version: u32,
    /// File path (relative to `kb_root`) → metadata.
    #[serde(default)]
    pub files: HashMap<String, FileEntry>,
}

/// Manifest manager. Reads/writes `kb/INDEX.toml`.
#[derive(Debug)]
pub struct KbManifest {
    /// Root directory containing the kb/ tree.
    kb_root: PathBuf,
    /// In-memory cached state.
    data: ManifestData,
    /// Held-open sentinel file with an exclusive advisory lock. Released
    /// automatically when this `KbManifest` is dropped.
    #[allow(dead_code)]
    lock_file: fs::File,
}

impl KbManifest {
    /// Open or create a manifest at `<kb_root>/INDEX.toml`.
    ///
    /// If the file doesn't exist, returns an empty manifest at the current
    /// `MANIFEST_VERSION` (a valid first-run state). Errors only on parse
    /// failure or I/O errors that aren't `NotFound`.
    pub fn open(kb_root: impl Into<PathBuf>) -> AlzinaResult<Self> {
        let kb_root = kb_root.into();

        // Validate root: must exist and be a directory.
        let root_meta = fs::metadata(&kb_root).map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("kb_root {} not accessible: {}", kb_root.display(), e),
                degraded: true,
                degradation_reason: Some(format!("kb_root metadata error: {}", e)),
            })
        })?;
        if !root_meta.is_dir() {
            return Err(AlzinaError::Search(SearchDetail {
                message: format!("kb_root {} is not a directory", kb_root.display()),
                degraded: true,
                degradation_reason: Some("kb_root is not a directory".into()),
            }));
        }

        // P1#6: acquire an advisory exclusive lock on a sentinel file
        // before touching the manifest. Two writers (e.g. the watcher
        // and a CLI reindex) racing on `save()` would otherwise risk
        // tempfile collisions or interleaved renames.
        let lock_path = kb_root.join(MANIFEST_LOCK_FILE);
        let lock_file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!(
                        "failed to open kb manifest lock {}: {}",
                        lock_path.display(),
                        e
                    ),
                    degraded: true,
                    degradation_reason: Some(format!("manifest lock open error: {}", e)),
                })
            })?;
        lock_file.try_lock_exclusive().map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: "kb manifest locked by another process".to_string(),
                degraded: true,
                degradation_reason: Some(format!(
                    "manifest lock {} held by another process: {}",
                    lock_path.display(),
                    e
                )),
            })
        })?;

        // Holding the exclusive lock means no other writer is mid-save,
        // so any leftover `.INDEX.toml.<uuid>.tmp` files are orphans
        // from a crashed save. Sweep them so the directory stays tidy.
        cleanup_orphan_tempfiles(&kb_root);

        let manifest_path = kb_root.join(MANIFEST_FILE);
        let raw = match fs::read_to_string(&manifest_path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self {
                    kb_root,
                    data: ManifestData {
                        version: MANIFEST_VERSION,
                        files: HashMap::new(),
                    },
                    lock_file,
                });
            }
            Err(e) => {
                return Err(AlzinaError::Search(SearchDetail {
                    message: format!("failed to read manifest {}: {}", manifest_path.display(), e),
                    degraded: true,
                    degradation_reason: Some(format!("manifest read error: {}", e)),
                }));
            }
        };

        let mut data: ManifestData = toml::from_str(&raw).map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!(
                    "failed to parse manifest {}: {}",
                    manifest_path.display(),
                    e
                ),
                degraded: true,
                degradation_reason: Some(format!("manifest parse error: {}", e)),
            })
        })?;

        if data.version == 0 {
            // Treat unset/legacy as the current version — first writers
            // may have produced files without explicit version=1.
            data.version = MANIFEST_VERSION;
        }

        if data.version > MANIFEST_VERSION {
            return Err(AlzinaError::Search(SearchDetail {
                message: format!(
                    "manifest version {} newer than supported {}",
                    data.version, MANIFEST_VERSION
                ),
                degraded: true,
                degradation_reason: Some(format!(
                    "manifest version {} newer than supported {} — refusing to load",
                    data.version, MANIFEST_VERSION
                )),
            }));
        }

        Ok(Self {
            kb_root,
            data,
            lock_file,
        })
    }

    /// Path to the manifest file: `<kb_root>/INDEX.toml`.
    pub fn path(&self) -> PathBuf {
        self.kb_root.join(MANIFEST_FILE)
    }

    /// Compute SHA-256 of file contents (hex). Streams the file so large
    /// inputs don't blow the heap.
    pub fn hash_file(path: &Path) -> AlzinaResult<String> {
        let mut file = fs::File::open(path).map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("failed to open {} for hashing: {}", path.display(), e),
                degraded: true,
                degradation_reason: Some(format!("hash open error: {}", e)),
            })
        })?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf).map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("failed to read {} for hashing: {}", path.display(), e),
                    degraded: true,
                    degradation_reason: Some(format!("hash read error: {}", e)),
                })
            })?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Whether `relative_path` (under `kb_root`) needs reindexing.
    ///
    /// Returns `true` if:
    ///   - the file is not in the manifest, OR
    ///   - the file's current SHA-256 differs from the manifest's recorded hash.
    ///
    /// Returns `false` only when the manifest's hash matches the on-disk
    /// hash. Errors on I/O failure when reading the file.
    pub fn needs_reindex(&self, relative_path: &str) -> AlzinaResult<bool> {
        let entry = match self.data.files.get(relative_path) {
            Some(e) => e,
            None => return Ok(true),
        };
        let absolute = self.kb_root.join(relative_path);
        let current = Self::hash_file(&absolute)?;
        Ok(current != entry.content_hash)
    }

    /// Mark a file as indexed. Records the supplied SHA-256, the chunk
    /// count, and the current UTC timestamp. Updates the in-memory cache;
    /// call [`save`](Self::save) to persist.
    pub fn mark_indexed(&mut self, relative_path: &str, content_hash: String, chunk_count: usize) {
        let entry = FileEntry {
            content_hash,
            last_indexed: Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            chunk_count,
        };
        self.data.files.insert(relative_path.to_string(), entry);
    }

    /// List files that exist in the manifest but NOT on disk under
    /// `kb_root`. These need their vectors deleted from the index.
    ///
    /// Walks `<kb_root>/` recursively for `.md` files using stdlib only.
    /// Skips dotfiles/dotdirs (e.g. `.git/`) and the manifest itself.
    pub fn list_removed(&self) -> AlzinaResult<Vec<String>> {
        let on_disk = collect_kb_files(&self.kb_root)?;
        let mut removed: Vec<String> = self
            .data
            .files
            .keys()
            .filter(|k| !on_disk.contains(k.as_str()))
            .cloned()
            .collect();
        removed.sort();
        Ok(removed)
    }

    /// List files that exist on disk under `kb_root` but are NOT in the
    /// manifest. These need a first-time index pass.
    ///
    /// Walks `<kb_root>/` recursively for `.md` files using stdlib only.
    /// Skips dotfiles/dotdirs (e.g. `.git/`) and the manifest itself.
    pub fn list_new(&self) -> AlzinaResult<Vec<String>> {
        let on_disk = collect_kb_files(&self.kb_root)?;
        let mut new_files: Vec<String> = on_disk
            .into_iter()
            .filter(|p| !self.data.files.contains_key(p))
            .collect();
        new_files.sort();
        Ok(new_files)
    }

    /// List files in the manifest that need reindexing (the on-disk hash
    /// differs from the recorded hash). Does NOT include never-indexed
    /// files — that's the indexer's job to discover via a directory walk.
    ///
    /// Files listed in the manifest but missing from disk are skipped here
    /// (use [`list_removed`](Self::list_removed) for those).
    pub fn list_stale(&self) -> AlzinaResult<Vec<String>> {
        let mut stale = Vec::new();
        for (rel, entry) in &self.data.files {
            let absolute = self.kb_root.join(rel);
            match fs::metadata(&absolute) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(AlzinaError::Search(SearchDetail {
                        message: format!(
                            "failed to stat {} while listing stale: {}",
                            absolute.display(),
                            e
                        ),
                        degraded: true,
                        degradation_reason: Some(format!("stat error: {}", e)),
                    }));
                }
            }
            let current = Self::hash_file(&absolute)?;
            if current != entry.content_hash {
                stale.push(rel.clone());
            }
        }
        stale.sort();
        Ok(stale)
    }

    /// Remove a file's entry from the manifest. Used after vectors are
    /// deleted from the index.
    pub fn remove(&mut self, relative_path: &str) {
        self.data.files.remove(relative_path);
    }

    /// Persist the manifest to `INDEX.toml`. Atomic via tempfile + rename
    /// so a crash mid-write can't leave a corrupt file.
    pub fn save(&self) -> AlzinaResult<()> {
        // Ensure version is set on save (defensive — `open` already
        // backfills, but constructing manually shouldn't write 0).
        let mut to_write = self.data.clone();
        if to_write.version == 0 {
            to_write.version = MANIFEST_VERSION;
        }

        let serialised = toml::to_string_pretty(&to_write).map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("failed to serialise manifest: {}", e),
                degraded: true,
                degradation_reason: Some(format!("manifest serialise error: {}", e)),
            })
        })?;

        // P1#6: each save uses a fresh per-attempt tempfile so two
        // writers in the same kb_root can't stomp on each other's
        // partial output. The advisory lock acquired in `open()` keeps
        // the cross-process invariant; the random suffix is belt-and-
        // braces in case a future API ever permits multiple in-process
        // saves to overlap.
        let tmp_name = format!(
            "{}{}{}",
            MANIFEST_TMP_PREFIX,
            Uuid::new_v4().simple(),
            MANIFEST_TMP_SUFFIX,
        );
        let tmp_path = self.kb_root.join(&tmp_name);
        let final_path = self.path();

        // Write to tempfile + fsync, then atomic rename.
        {
            let mut tmp = fs::File::create(&tmp_path).map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!(
                        "failed to create manifest tempfile {}: {}",
                        tmp_path.display(),
                        e
                    ),
                    degraded: true,
                    degradation_reason: Some(format!("tempfile create error: {}", e)),
                })
            })?;
            tmp.write_all(serialised.as_bytes()).map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("failed to write manifest tempfile: {}", e),
                    degraded: true,
                    degradation_reason: Some(format!("tempfile write error: {}", e)),
                })
            })?;
            tmp.sync_all().map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("failed to fsync manifest tempfile: {}", e),
                    degraded: true,
                    degradation_reason: Some(format!("tempfile fsync error: {}", e)),
                })
            })?;
        }

        fs::rename(&tmp_path, &final_path).map_err(|e| {
            // Best-effort cleanup of the orphaned tempfile.
            let _ = fs::remove_file(&tmp_path);
            AlzinaError::Search(SearchDetail {
                message: format!(
                    "failed to rename manifest tempfile to {}: {}",
                    final_path.display(),
                    e
                ),
                degraded: true,
                degradation_reason: Some(format!("manifest rename error: {}", e)),
            })
        })?;

        Ok(())
    }

    /// Read-only access to the underlying data (for inspection/testing).
    pub fn data(&self) -> &ManifestData {
        &self.data
    }
}

/// Best-effort cleanup of orphan save tempfiles in `kb_root`. Any file
/// whose name starts with `MANIFEST_TMP_PREFIX` and ends with
/// `MANIFEST_TMP_SUFFIX` is from a crashed prior save (the holder of
/// the exclusive lock is the only writer, so live tempfiles can't
/// exist while we hold it). I/O errors are logged at debug and
/// otherwise ignored — a stale tempfile is annoying, not load-bearing.
fn cleanup_orphan_tempfiles(kb_root: &Path) {
    let entries = match fs::read_dir(kb_root) {
        Ok(it) => it,
        Err(e) => {
            tracing::debug!(
                target: "alzina_search::manifest",
                kb_root = %kb_root.display(),
                error = %e,
                "skipping orphan tempfile sweep: read_dir failed"
            );
            return;
        }
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with(MANIFEST_TMP_PREFIX) && name_str.ends_with(MANIFEST_TMP_SUFFIX) {
            let path = entry.path();
            if let Err(e) = fs::remove_file(&path) {
                tracing::debug!(
                    target: "alzina_search::manifest",
                    path = %path.display(),
                    error = %e,
                    "failed to remove orphan manifest tempfile"
                );
            }
        }
    }
}

/// Walk `kb_root` recursively collecting `.md` file paths relative to
/// `kb_root`, with forward-slash separators. Skips dotfiles/dotdirs and
/// the manifest itself.
fn collect_kb_files(kb_root: &Path) -> AlzinaResult<std::collections::HashSet<String>> {
    let mut out = std::collections::HashSet::new();
    walk_dir(kb_root, kb_root, &mut out)?;
    Ok(out)
}

fn walk_dir(
    root: &Path,
    dir: &Path,
    out: &mut std::collections::HashSet<String>,
) -> AlzinaResult<()> {
    let entries = fs::read_dir(dir).map_err(|e| {
        AlzinaError::Search(SearchDetail {
            message: format!("failed to read_dir {}: {}", dir.display(), e),
            degraded: true,
            degradation_reason: Some(format!("read_dir error: {}", e)),
        })
    })?;
    for entry in entries {
        let entry = entry.map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("dir-entry error in {}: {}", dir.display(), e),
                degraded: true,
                degradation_reason: Some(format!("dir-entry error: {}", e)),
            })
        })?;
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with('.') {
            continue; // skip dotfiles + dotdirs
        }
        let file_type = entry.file_type().map_err(|e| {
            AlzinaError::Search(SearchDetail {
                message: format!("file_type error for {}: {}", path.display(), e),
                degraded: true,
                degradation_reason: Some(format!("file_type error: {}", e)),
            })
        })?;
        if file_type.is_symlink() {
            tracing::debug!(target: "alzina_search::manifest", path = %path.display(), "skipping symlink in kb walk");
            continue;
        }
        if file_type.is_dir() {
            walk_dir(root, &path, out)?;
        } else if file_type.is_file() {
            // Only consider Markdown.
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                continue;
            }
            // Skip the manifest itself just in case it ever ends in .md.
            if name_str == MANIFEST_FILE {
                continue;
            }
            let rel = path.strip_prefix(root).map_err(|e| {
                AlzinaError::Search(SearchDetail {
                    message: format!("strip_prefix error: {}", e),
                    degraded: true,
                    degradation_reason: Some(format!("strip_prefix error: {}", e)),
                })
            })?;
            // Forward-slash join — portable across platforms.
            let mut parts = Vec::new();
            for comp in rel.components() {
                parts.push(comp.as_os_str().to_string_lossy().into_owned());
            }
            out.insert(parts.join("/"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn write(path: &Path, body: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, body).unwrap();
    }

    #[test]
    fn open_creates_empty_manifest_when_file_missing() {
        let dir = tempdir().unwrap();
        let m = KbManifest::open(dir.path()).unwrap();
        assert!(m.data().files.is_empty());
        assert_eq!(m.data().version, MANIFEST_VERSION);
        // No file written yet.
        assert!(!dir.path().join(MANIFEST_FILE).exists());
    }

    #[test]
    fn save_then_open_round_trips_data() {
        let dir = tempdir().unwrap();
        {
            let mut m = KbManifest::open(dir.path()).unwrap();
            m.mark_indexed("a.md", "hash-a".into(), 3);
            m.mark_indexed("b.md", "hash-b".into(), 7);
            m.save().unwrap();
        }
        let m = KbManifest::open(dir.path()).unwrap();
        assert_eq!(m.data().files.len(), 2);
        assert_eq!(m.data().files["a.md"].content_hash, "hash-a");
        assert_eq!(m.data().files["a.md"].chunk_count, 3);
        assert_eq!(m.data().files["b.md"].content_hash, "hash-b");
        assert_eq!(m.data().files["b.md"].chunk_count, 7);
    }

    #[test]
    fn mark_indexed_records_hash_and_count() {
        let dir = tempdir().unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("foo.md", "deadbeef".into(), 5);
        let entry = &m.data().files["foo.md"];
        assert_eq!(entry.content_hash, "deadbeef");
        assert_eq!(entry.chunk_count, 5);
        assert!(!entry.last_indexed.is_empty());
        // Looks like an ISO-8601 stamp.
        assert!(entry.last_indexed.contains('T'));
        assert!(entry.last_indexed.ends_with('Z'));
    }

    #[test]
    fn needs_reindex_returns_true_for_unknown_file() {
        let dir = tempdir().unwrap();
        write(&dir.path().join("new.md"), "hello");
        let m = KbManifest::open(dir.path()).unwrap();
        assert!(m.needs_reindex("new.md").unwrap());
    }

    #[test]
    fn needs_reindex_returns_false_for_unchanged_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("doc.md");
        write(&path, "alpha");
        let hash = KbManifest::hash_file(&path).unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("doc.md", hash, 1);
        assert!(!m.needs_reindex("doc.md").unwrap());
    }

    #[test]
    fn needs_reindex_returns_true_for_changed_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("doc.md");
        write(&path, "alpha");
        let original = KbManifest::hash_file(&path).unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("doc.md", original, 1);
        // Mutate file.
        write(&path, "beta-different");
        assert!(m.needs_reindex("doc.md").unwrap());
    }

    #[test]
    fn list_removed_finds_manifest_only_files() {
        let dir = tempdir().unwrap();
        let a = dir.path().join("a.md");
        let b = dir.path().join("b.md");
        write(&a, "A");
        write(&b, "B");
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("a.md", KbManifest::hash_file(&a).unwrap(), 1);
        m.mark_indexed("b.md", KbManifest::hash_file(&b).unwrap(), 1);
        // Delete b.md from disk.
        fs::remove_file(&b).unwrap();
        let removed = m.list_removed().unwrap();
        assert_eq!(removed, vec!["b.md".to_string()]);
    }

    #[test]
    fn list_new_finds_disk_only_files() {
        let dir = tempdir().unwrap();
        let a = dir.path().join("a.md");
        let b = dir.path().join("nested/b.md");
        let c = dir.path().join("c.md");
        write(&a, "A");
        write(&b, "B");
        write(&c, "C");
        let mut m = KbManifest::open(dir.path()).unwrap();
        // Only `a.md` is in the manifest; `nested/b.md` and `c.md` are new.
        m.mark_indexed("a.md", KbManifest::hash_file(&a).unwrap(), 1);
        let new_files = m.list_new().unwrap();
        assert_eq!(
            new_files,
            vec!["c.md".to_string(), "nested/b.md".to_string()]
        );
    }

    #[test]
    fn list_stale_finds_changed_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("c.md");
        write(&path, "first");
        let original = KbManifest::hash_file(&path).unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("c.md", original, 1);
        write(&path, "second-different");
        let stale = m.list_stale().unwrap();
        assert_eq!(stale, vec!["c.md".to_string()]);
    }

    #[test]
    fn list_stale_excludes_unchanged_files() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("d.md");
        write(&path, "stable");
        let hash = KbManifest::hash_file(&path).unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("d.md", hash, 1);
        assert!(m.list_stale().unwrap().is_empty());
    }

    #[test]
    fn hash_file_is_stable() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("h.md");
        write(&p, "stable-content");
        let h1 = KbManifest::hash_file(&p).unwrap();
        let h2 = KbManifest::hash_file(&p).unwrap();
        assert_eq!(h1, h2);
        write(&p, "different-content");
        let h3 = KbManifest::hash_file(&p).unwrap();
        assert_ne!(h1, h3);
        // Hex-encoded SHA-256 is 64 chars.
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn save_is_atomic_via_tempfile() {
        let dir = tempdir().unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("x.md", "h".into(), 1);
        m.save().unwrap();
        assert!(dir.path().join(MANIFEST_FILE).exists());
        // No randomly-named tempfile should remain after a successful save.
        assert!(
            list_orphan_tempfiles(dir.path()).is_empty(),
            "save left tempfiles behind: {:?}",
            list_orphan_tempfiles(dir.path())
        );
    }

    /// Test helper: enumerate any leftover `.INDEX.toml.<uuid>.tmp`
    /// files under `kb_root`. Returns full paths.
    fn list_orphan_tempfiles(kb_root: &Path) -> Vec<PathBuf> {
        fs::read_dir(kb_root)
            .map(|it| {
                it.flatten()
                    .filter_map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if name.starts_with(MANIFEST_TMP_PREFIX)
                            && name.ends_with(MANIFEST_TMP_SUFFIX)
                        {
                            Some(e.path())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn manifest_with_unsupported_version_errors() {
        let dir = tempdir().unwrap();
        write(&dir.path().join(MANIFEST_FILE), "version = 999\n[files]\n");
        let err = KbManifest::open(dir.path()).expect_err("should reject newer version");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded, "version-too-new must surface as degraded");
                let reason = detail.degradation_reason.unwrap_or_default();
                assert!(
                    reason.to_lowercase().contains("version"),
                    "degradation reason should mention version, got: {reason}"
                );
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }

    #[test]
    fn remove_deletes_entry() {
        let dir = tempdir().unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        m.mark_indexed("z.md", "h".into(), 1);
        assert!(m.data().files.contains_key("z.md"));
        m.remove("z.md");
        assert!(!m.data().files.contains_key("z.md"));
    }

    #[cfg(unix)]
    #[test]
    fn walk_dir_skips_symlink_to_directory() {
        // A symlinked directory must not be recursed into — otherwise a
        // `kb/escape -> /etc` link would let the walker pick up
        // arbitrary `.md` files outside `kb_root`.
        let dir = tempdir().unwrap();
        // Create the legitimate kb tree.
        let inside = dir.path().join("real.md");
        write(&inside, "real");

        // Create a separate dir containing a `.md` and symlink it into kb.
        let outside = tempdir().unwrap();
        let outside_md = outside.path().join("escaped.md");
        write(&outside_md, "outside");
        let link = dir.path().join("escape");
        std::os::unix::fs::symlink(outside.path(), &link).unwrap();

        let m = KbManifest::open(dir.path()).unwrap();
        let new_files = m.list_new().unwrap();
        // Only the genuine file should be discovered.
        assert!(
            new_files.iter().any(|p| p == "real.md"),
            "expected real.md in {new_files:?}"
        );
        assert!(
            !new_files.iter().any(|p| p.contains("escape")),
            "walker followed a symlinked dir: {new_files:?}"
        );
        assert!(
            !new_files.iter().any(|p| p.contains("escaped.md")),
            "walker exfiltrated escaped.md: {new_files:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn walk_dir_skips_symlink_to_file() {
        // A symlinked file must also be skipped — same exfil concern at
        // file granularity.
        let dir = tempdir().unwrap();
        let real = dir.path().join("real.md");
        write(&real, "real");

        let outside = tempdir().unwrap();
        let target = outside.path().join("secret.md");
        write(&target, "secret");
        let link = dir.path().join("link.md");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let m = KbManifest::open(dir.path()).unwrap();
        let new_files = m.list_new().unwrap();
        assert!(
            new_files.iter().any(|p| p == "real.md"),
            "expected real.md in {new_files:?}"
        );
        assert!(
            !new_files.iter().any(|p| p == "link.md"),
            "walker followed a symlinked file: {new_files:?}"
        );
    }

    #[test]
    fn relative_paths_use_forward_slashes() {
        let dir = tempdir().unwrap();
        {
            let mut m = KbManifest::open(dir.path()).unwrap();
            m.mark_indexed("sub/dir/foo.md", "hash-fwd".into(), 2);
            m.save().unwrap();
        }
        let m2 = KbManifest::open(dir.path()).unwrap();
        assert!(m2.data().files.contains_key("sub/dir/foo.md"));
        assert_eq!(m2.data().files["sub/dir/foo.md"].content_hash, "hash-fwd");
    }

    #[test]
    fn concurrent_save_does_not_collide() {
        // P1#6: a single owner doing 50 successive saves must never
        // leave orphan tempfiles, and the final manifest must be the
        // last write we asked for.
        let dir = tempdir().unwrap();
        let mut m = KbManifest::open(dir.path()).unwrap();
        let mut last_hash = String::new();
        for i in 0..50 {
            last_hash = format!("hash-{i}-{}", Uuid::new_v4().simple());
            m.mark_indexed("doc.md", last_hash.clone(), i);
            m.save().unwrap();
        }
        // No orphan `.INDEX.toml.<uuid>.tmp` files.
        let orphans = list_orphan_tempfiles(dir.path());
        assert!(
            orphans.is_empty(),
            "tempfiles leaked across 50 saves: {orphans:?}"
        );
        // Final manifest reflects the last save.
        drop(m);
        let reopened = KbManifest::open(dir.path()).unwrap();
        assert_eq!(reopened.data().files["doc.md"].content_hash, last_hash);
        assert_eq!(reopened.data().files["doc.md"].chunk_count, 49);
    }

    #[test]
    fn second_open_fails_when_first_is_held() {
        let dir = tempdir().unwrap();
        let _m1 = KbManifest::open(dir.path()).expect("first open holds the lock");
        let err = KbManifest::open(dir.path()).expect_err("second open must fail while held");
        match err {
            AlzinaError::Search(detail) => {
                assert!(detail.degraded, "lock contention must surface as degraded");
                let reason = detail.degradation_reason.unwrap_or_default();
                assert!(
                    reason.to_lowercase().contains("lock"),
                    "degradation reason should mention lock, got: {reason}"
                );
                assert!(
                    detail.message.to_lowercase().contains("lock"),
                    "message should mention lock, got: {}",
                    detail.message
                );
            }
            other => panic!("expected AlzinaError::Search, got {other:?}"),
        }
    }

    #[test]
    fn lock_released_on_drop() {
        let dir = tempdir().unwrap();
        {
            let _m1 = KbManifest::open(dir.path()).unwrap();
            // _m1 dropped at end of scope — lock released.
        }
        // A subsequent open must succeed now that the lock is free.
        let _m2 = KbManifest::open(dir.path()).expect("lock should be released on drop");
    }

    #[test]
    fn open_cleans_up_orphan_tempfiles() {
        let dir = tempdir().unwrap();
        // Plant an orphan tempfile as if a prior save crashed mid-write.
        let orphan = dir.path().join(format!(
            "{}deadbeef{}",
            MANIFEST_TMP_PREFIX, MANIFEST_TMP_SUFFIX
        ));
        fs::write(&orphan, "garbage from a crashed save").unwrap();
        assert!(orphan.exists(), "precondition: orphan exists");

        let _m = KbManifest::open(dir.path()).unwrap();
        assert!(
            !orphan.exists(),
            "open should sweep orphan tempfile {}",
            orphan.display()
        );
    }
}
