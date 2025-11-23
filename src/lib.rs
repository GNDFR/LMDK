// --------------------------------------------------------------------------------
// |
// |    **PyO3 and Standard Library Imports**
// |
// |  - `pyo3::prelude::*`: Essential components for creating Python bindings from Rust.
// |  - `std::fs::File`, `std::io::{self, BufRead, Write}`: For file I/O operations.
// |  - `std::path::Path`: For handling filesystem paths.
// |  - `fnv::FnvHashSet`: A fast hash set for storing hashes of seen lines.
// |  - `aho_corasick::AhoCorasick`: For efficient multi-pattern string matching.
// |  - `std::collections::hash_map::DefaultHasher`, `std::hash::{Hash, Hasher}`: For hashing lines.
// |
// --------------------------------------------------------------------------------
use pyo3::prelude::*;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::Path;
use fnv::FnvHashSet;
use aho_corasick::AhoCorasick;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// --------------------------------------------------------------------------------
// |
// |    **DataCleanser Struct**
// |
// |  This struct is the core of the data cleaning functionality. It holds the
// |  configuration for the cleaning process and the state of the cleaned data.
// |
// --------------------------------------------------------------------------------
#[pyclass(name = "DataCleanser")]
struct DataCleanser {
    min_length: usize,
    toxic_keywords_automaton: AhoCorasick,
    seen_lines_hashes: FnvHashSet<u64>,
    cleaned_lines: Vec<String>,
}

// --------------------------------------------------------------------------------
// |
// |    **Python Methods for DataCleanser**
// |
// |  These methods are exposed to Python and provide the interface for the
// |  data cleaning functionality.
// |
// --------------------------------------------------------------------------------
#[pymethods]
impl DataCleanser {
    // --------------------------------------------------------------------------------
    // |  `new` - The constructor for the DataCleanser class.
    // --------------------------------------------------------------------------------
    #[new]
    #[pyo3(signature = (min_length=20, toxic_keywords=None))]
    fn new(min_length: usize, toxic_keywords: Option<Vec<String>>) -> PyResult<Self> {
        let patterns = toxic_keywords.unwrap_or_else(Vec::new);
        let automaton = AhoCorasick::new(patterns).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(DataCleanser {
            min_length,
            toxic_keywords_automaton: automaton,
            seen_lines_hashes: FnvHashSet::default(),
            cleaned_lines: Vec::new(),
        })
    }

    // --------------------------------------------------------------------------------
    // |  `process_file` - Processes a file, cleaning and deduplicating the lines.
    // --------------------------------------------------------------------------------
    fn process_file(&mut self, filepath: &str) -> PyResult<usize> {
        let path = Path::new(filepath);
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            let main_content = line.split('#').next().unwrap_or("").trim();
            let cleaned_content = main_content.replace(' ', " ");

            if cleaned_content.chars().count() < self.min_length {
                continue;
            }

            let lowercased = cleaned_content.to_lowercase();
            if self.toxic_keywords_automaton.patterns_len() > 0 && self.toxic_keywords_automaton.is_match(&lowercased) {
                continue;
            }

            let mut hasher = DefaultHasher::new();
            lowercased.hash(&mut hasher);
            let line_hash = hasher.finish();

            if self.seen_lines_hashes.insert(line_hash) {
                self.cleaned_lines.push(cleaned_content);
            }
        }
        Ok(self.cleaned_lines.len())
    }

    // --------------------------------------------------------------------------------
    // |  `save_cleaned_to_file` - Saves the cleaned lines to a file.
    // --------------------------------------------------------------------------------
    fn save_cleaned_to_file(&self, output_path: &str) -> PyResult<()> {
        let path = Path::new(output_path);
        let mut file = File::create(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

        for line in &self.cleaned_lines {
            writeln!(file, "{}", line).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        }

        Ok(())
    }

    // --------------------------------------------------------------------------------
    // |  `count` - Returns the number of unique lines found.
    // --------------------------------------------------------------------------------
    #[getter]
    fn count(&self) -> PyResult<usize> {
        Ok(self.cleaned_lines.len())
    }
}

#[pymodule]
fn rust_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DataCleanser>()?;
    Ok(())
}
