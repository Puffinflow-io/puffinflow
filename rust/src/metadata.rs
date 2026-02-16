/// Compact state metadata entry.
///
/// ~32 bytes per state vs ~400+ bytes for the Python dataclass equivalent.

/// Status constants matching Python side.
pub const STATUS_PENDING: u8 = 0;
pub const STATUS_RUNNING: u8 = 1;
pub const STATUS_COMPLETED: u8 = 2;
pub const STATUS_FAILED: u8 = 3;

/// Status name lookup.
pub const STATUS_NAMES: [&str; 4] = ["pending", "running", "completed", "failed"];

#[derive(Clone, Debug)]
pub struct StateMetadataEntry {
    pub status: u8,
    pub attempts: u32,
    pub max_retries: u32,
    pub last_execution: f64,
    pub last_success: f64,
    pub priority: i32,
}

impl StateMetadataEntry {
    pub fn new(priority: i32, max_retries: u32) -> Self {
        Self {
            status: STATUS_PENDING,
            attempts: 0,
            max_retries,
            last_execution: 0.0,
            last_success: 0.0,
            priority,
        }
    }

    pub fn reset(&mut self) {
        self.status = STATUS_PENDING;
        self.attempts = 0;
        self.last_execution = 0.0;
        self.last_success = 0.0;
    }

    /// Get status name string.
    pub fn status_name(&self) -> &'static str {
        let idx = self.status as usize;
        if idx < STATUS_NAMES.len() {
            STATUS_NAMES[idx]
        } else {
            "pending"
        }
    }
}
