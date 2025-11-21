use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;
use fnv::FnvHashSet;
use std::fs::File; // Rust 파일 시스템 모듈
use std::io::{BufRead, BufReader}; // 버퍼링된 읽기 도구

// 1. DataCleanser 구조체
#[pyclass]
pub struct DataCleanser {
    pub seen_texts: FnvHashSet<String>,
    pub toxic_keywords: FnvHashSet<String>, // 형님이 추가하신 기능 유지!
}

// 2. DataCleanser 메소드 구현
#[pymethods]
impl DataCleanser {
    #[new]
    fn new() -> Self {
        let mut toxic_keywords = FnvHashSet::default();
        toxic_keywords.insert("badword1".to_string());
        toxic_keywords.insert("badword2".to_string());
        toxic_keywords.insert("offensive_term".to_string());

        DataCleanser {
            seen_texts: FnvHashSet::default(),
            toxic_keywords,
        }
    }

    /// 텍스트를 정규화하고 중복을 제거합니다.
    pub fn clean_text(&mut self, text: String) -> PyResult<Option<String>> {
        // 1. 정규화 (Normalization)
        let normalized_key = text
            .replace(' ', " ")
            .trim()
            .to_lowercase();

        // 2. 필터링 (Filtering) - 문자 수 기준 (20자 미만)
        const MIN_LENGTH: usize = 20;
        if normalized_key.chars().count() < MIN_LENGTH {
            // eprintln!("Filtered length: {}", normalized_key); // 디버깅 필요시 주석 해제
            return Ok(None);
        }

        // 유해 콘텐츠 필터링 (형님이 만드신 로직!)
        for keyword in &self.toxic_keywords {
            if normalized_key.contains(keyword) {
                // eprintln!("Filtered toxic: {}", normalized_key);
                return Ok(None);
            }
        }

        // 3. 중복 체크 (Deduplication)
        if self.seen_texts.contains(&normalized_key) {
            Ok(None)
        } else {
            self.seen_texts.insert(normalized_key);
            Ok(Some(text)) // 원본 반환
        }
    }

    // 🔥 [핵심] 고성능 파일 처리 파이프라인
    // Python 파일 객체 대신, 파일 경로(path)를 받아서 Rust가 직접 엽니다.
    pub fn process_file(&mut self, path: String) -> PyResult<usize> {
        // Rust의 File I/O를 사용하면 Python GIL과 상관없이 엄청나게 빠릅니다.
        // 또한 BufReader를 사용하여 메모리를 조금씩만 사용합니다 (Streaming).

        let file = File::open(path)?; // 파일 열기 (실패시 에러 반환)
        let reader = BufReader::new(file);
        let mut processed_lines = 0;

        for line in reader.lines() {
            // Rust Result 처리
            let text = match line {
                Ok(t) => t,
                Err(e) => return Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
            };

            // clean_text 호출
            // unwrap_or(None).is_some() 패턴으로 유효한 문장만 카운트
            if self.clean_text(text).unwrap_or(None).is_some() {
                processed_lines += 1;
            }
        }

        Ok(processed_lines)
    }

    // 추적 개수
    #[getter]
    pub fn count(&self) -> PyResult<usize> {
        Ok(self.seen_texts.len())
    }
}

// 3. 모듈 진입점
#[pymodule]
fn rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DataCleanser>()?;
    Ok(())
}
