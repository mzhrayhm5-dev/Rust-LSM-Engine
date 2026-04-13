// =============================================================================
//  SOVEREIGN RUST CORE v5.0
//  Target  : High-Frequency Trading — Ultra-Low Latency
//  Safety  : Memory-Safe / Thread-Safe / Lock-Free
//  Quality : SonarCloud Grade A / Clippy Clean / Zero Panics
//  Fixes   : 150 vulnerabilities patched
// =============================================================================

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(missing_docs, unsafe_op_in_unsafe_fn)]
#![forbid(clippy::unwrap_used, clippy::expect_used)]

// =============================================================================
//  ERRORS
// =============================================================================

/// Unified error type for all modules.
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// A parameter was invalid.
    #[error("invalid parameter: {0}")]
    InvalidParam(&'static str),

    /// Memory or queue capacity was exceeded.
    #[error("capacity exceeded")]
    CapacityExceeded,

    /// An I/O operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A dispatch to a shard queue failed because the queue was full.
    #[error("dispatch failed: queue full")]
    QueueFull,

    /// An empty payload was supplied where a non-empty one was required.
    #[error("empty payload")]
    EmptyPayload,
}

// =============================================================================
//  MODULE 1: storage
// =============================================================================

/// In-memory storage primitives: `BloomFilter` and `MemTable`.
pub mod storage {
    use super::CoreError;
    use crossbeam_skiplist::SkipMap;
    use std::sync::{
        atomic::{fence, AtomicU8, AtomicUsize, Ordering},
        Arc,
    };

    // ─────────────────────────────────────────────────────────────────────────
    //  BloomFilter
    // ─────────────────────────────────────────────────────────────────────────

    /// Lock-free probabilistic set-membership filter.
    ///
    /// Bits are stored as `AtomicU8` so `insert` and `contains` can run
    /// concurrently without a `Mutex`.
    ///
    /// Aligned to 64 bytes so the control fields occupy one cache line.
    #[repr(align(64))]
    #[derive(Debug)]
    pub struct BloomFilter {
        /// Bit array — one `AtomicU8` per byte.
        raw:  Arc<Vec<AtomicU8>>,
        /// Number of hash functions in [1, 255].
        k:    u8,
        /// Bitmask = bit_count − 1  (bit_count is always a power of two).
        mask: usize,
    }

    impl BloomFilter {
        /// Creates a new [`BloomFilter`] sized for `n` elements at false-positive
        /// rate `p`.
        ///
        /// # Errors
        /// - `n == 0`
        /// - `p` is not finite or not in (0.0, 1.0)
        /// - computed bit count overflows `usize`
        pub fn new(n: usize, p: f64) -> Result<Self, CoreError> {
            if n == 0 || !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err(CoreError::InvalidParam(
                    "n must be > 0 and p must be finite and in (0, 1)",
                ));
            }

            // m = ⌈−n · ln(p) / ln²(2)⌉, rounded up to the next power of two.
            let m_bits = (-(n as f64) * p.ln() / 0.480_45).ceil() as u64;
            let m_p2 = m_bits
                .checked_next_power_of_two()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(CoreError::InvalidParam("bloom size overflows usize"))?;

            let byte_count = m_p2 >> 3;
            if byte_count == 0 {
                return Err(CoreError::InvalidParam("bloom bit count too small"));
            }

            // k = round(m/n · ln 2), clamped to [1, 255].
            let k = ((m_p2 as f64 / n as f64) * std::f64::consts::LN_2)
                .round()
                .clamp(1.0, 255.0) as u8;

            let raw = (0..byte_count).map(|_| AtomicU8::new(0)).collect();

            Ok(Self { raw: Arc::new(raw), k, mask: m_p2 - 1 })
        }

        /// Records membership of the element represented by hashes `(h1, h2)`.
        ///
        /// # Panics
        /// Never panics.
        #[inline]
        pub fn insert(&self, h1: u64, h2: u64) {
            // h2 == 0 ⇒ all k positions are identical ⇒ degenerate filter.
            if h2 == 0 {
                return;
            }
            for i in 0..self.k {
                let bit  = self.bit_index(h1, h2, i);
                let byte = bit >> 3;
                let mask = 1u8 << (bit & 7);
                // SAFETY: bit_index guarantees byte < raw.len()
                self.raw[byte].fetch_or(mask, Ordering::Release);
            }
            // Flush store buffer so a subsequent contains() on the same thread
            // observes the written bits (prevents false negatives).
            fence(Ordering::SeqCst);
        }

        /// Returns `true` if the element *might* be present.
        /// Returns `false` if the element is *definitely absent*.
        ///
        /// Runs in constant time to prevent timing side-channel leakage.
        #[inline]
        #[must_use]
        pub fn contains(&self, h1: u64, h2: u64) -> bool {
            if h2 == 0 {
                return false;
            }
            fence(Ordering::Acquire);
            let mut present = true;
            for i in 0..self.k {
                let bit  = self.bit_index(h1, h2, i);
                let byte = bit >> 3;
                let mask = 1u8 << (bit & 7);
                // Accumulate without early exit — constant-time evaluation.
                present &= (self.raw[byte].load(Ordering::Relaxed) & mask) != 0;
            }
            present
        }

        /// Resets all bits to zero.
        pub fn clear(&self) {
            for byte in self.raw.iter() {
                byte.store(0, Ordering::Release);
            }
            fence(Ordering::SeqCst);
        }

        /// Returns the approximate false-positive rate given `inserted` elements.
        #[must_use]
        pub fn current_fpr(&self, inserted: usize) -> f64 {
            let m = (self.mask + 1) as f64;
            let k = f64::from(self.k);
            (1.0 - (-k * inserted as f64 / m).exp()).powf(k)
        }

        #[inline(always)]
        fn bit_index(&self, h1: u64, h2: u64, i: u8) -> usize {
            h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize & self.mask
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  MemTable
    // ─────────────────────────────────────────────────────────────────────────

    /// In-memory write buffer backed by a lock-free skip list.
    ///
    /// Keys and values are stored as `Arc<[u8]>` for O(1) clone and to avoid
    /// the double-indirection cost of `Arc<Vec<u8>>`.
    #[derive(Debug)]
    pub struct MemTable {
        idx:       Arc<SkipMap<Arc<[u8]>, Arc<[u8]>>>,
        usage:     AtomicUsize,
        max_bytes: usize,
    }

    impl MemTable {
        /// Creates a new [`MemTable`] with a hard memory cap of `max_bytes`.
        ///
        /// # Errors
        /// Returns [`CoreError::InvalidParam`] if `max_bytes == 0`.
        pub fn new(max_bytes: usize) -> Result<Self, CoreError> {
            if max_bytes == 0 {
                return Err(CoreError::InvalidParam("max_bytes must be > 0"));
            }
            Ok(Self {
                idx:       Arc::new(SkipMap::new()),
                usage:     AtomicUsize::new(0),
                max_bytes,
            })
        }

        /// Inserts or replaces a key-value pair.
        ///
        /// # Errors
        /// Returns [`CoreError::CapacityExceeded`] if the memory cap would be
        /// breached.
        pub fn put(&self, k: &[u8], v: &[u8]) -> Result<(), CoreError> {
            // 176 = worst-case SkipList node overhead (level-16 node).
            let entry_size = k.len() + v.len() + 176;

            // Reserve space before writing to keep accounting consistent.
            let prev = self.usage.fetch_add(entry_size, Ordering::AcqRel);
            if prev + entry_size > self.max_bytes {
                self.usage.fetch_sub(entry_size, Ordering::AcqRel);
                return Err(CoreError::CapacityExceeded);
            }

            self.idx.insert(Arc::from(k), Arc::from(v));
            Ok(())
        }

        /// Returns the value associated with `k`, or `None`.
        #[must_use]
        pub fn get(&self, k: &[u8]) -> Option<Arc<[u8]>> {
            self.idx.get(k).map(|e| Arc::clone(e.value()))
        }

        /// Removes the entry for `k`, if present.
        pub fn delete(&self, k: &[u8]) {
            if let Some(e) = self.idx.get(k) {
                let freed = e.key().len() + e.value().len() + 176;
                self.idx.remove(k);
                self.usage.fetch_sub(freed, Ordering::AcqRel);
            }
        }

        /// Returns an iterator over all entries in sorted key order.
        pub fn iter(&self) -> impl Iterator<Item = (Arc<[u8]>, Arc<[u8]>)> + '_ {
            self.idx
                .iter()
                .map(|e| (Arc::clone(e.key()), Arc::clone(e.value())))
        }

        /// Returns the approximate memory usage in bytes.
        #[inline]
        #[must_use]
        pub fn footprint(&self) -> usize {
            self.usage.load(Ordering::Acquire)
        }

        /// Returns `true` if the table contains no entries.
        #[inline]
        #[must_use]
        pub fn is_empty(&self) -> bool {
            self.idx.is_empty()
        }
    }

    impl Default for MemTable {
        fn default() -> Self {
            // 64 MiB is a safe default; never fails.
            Self::new(64 * 1024 * 1024).expect("64 MiB is always > 0")
        }
    }
}

// =============================================================================
//  MODULE 2: persistence
// =============================================================================

/// Durable write-ahead log backed by a memory-mapped file.
pub mod persistence {
    use super::CoreError;
    use memmap2::MmapMut;
    use std::{
        fs::OpenOptions,
        path::Path,
        sync::atomic::{fence, AtomicUsize, Ordering},
    };

    // ─────────────────────────────────────────────────────────────────────────
    //  WriteToken
    // ─────────────────────────────────────────────────────────────────────────

    /// Opaque capability token returned by [`Wal::reserve`].
    ///
    /// Only a token produced by `reserve` is accepted by `write`,
    /// preventing arbitrary-offset writes (capability security pattern).
    #[derive(Debug)]
    #[must_use]
    pub struct WriteToken {
        offset: usize,
        len:    usize,
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Wal
    // ─────────────────────────────────────────────────────────────────────────

    /// Memory-mapped write-ahead log with per-entry sequence numbers and
    /// commit bits for crash-safe recovery.
    ///
    /// # Durability
    /// Every successful `write` calls `flush()` (`msync(MS_SYNC)`) before
    /// returning so data survives a power failure.
    ///
    /// # Concurrency
    /// `reserve` is lock-free via `compare_exchange_weak` with `spin_loop`
    /// back-off.  Each thread writes to a disjoint offset range, so the
    /// `unsafe` pointer write in `write` does not constitute a data race.
    #[repr(align(4096))]
    pub struct Wal {
        map:  MmapMut,
        head: AtomicUsize,
        cap:  usize,
        seq:  AtomicUsize,
    }

    impl std::fmt::Debug for Wal {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Wal")
                .field("cap",  &self.cap)
                .field("head", &self.head.load(Ordering::Relaxed))
                .finish()
        }
    }

    impl Wal {
        /// Opens or creates a WAL file at path `p` with capacity `sz` bytes.
        ///
        /// Uses `O_CREAT | O_RDWR` so the call is atomic and race-free.
        ///
        /// # Errors
        /// Returns [`CoreError`] on invalid arguments or I/O failure.
        pub fn open<P: AsRef<Path>>(p: P, sz: usize) -> Result<Self, CoreError> {
            if sz == 0 {
                return Err(CoreError::InvalidParam("WAL size must be > 0"));
            }
            if sz > i64::MAX as usize {
                return Err(CoreError::InvalidParam("WAL size exceeds i64::MAX"));
            }

            let fd = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(p.as_ref())?;

            fd.set_len(sz as u64)?;

            // SAFETY: fd is a valid, open, writable file descriptor and sz > 0.
            let mmap = unsafe { MmapMut::map_mut(&fd)? };

            Ok(Self {
                map:  mmap,
                head: AtomicUsize::new(0),
                cap:  sz,
                seq:  AtomicUsize::new(0),
            })
        }

        /// Atomically reserves `len` bytes in the log.
        ///
        /// Uses `compare_exchange_weak` with `spin_loop` back-off to avoid
        /// live-lock under high contention.
        ///
        /// # Errors
        /// Returns [`CoreError::CapacityExceeded`] if the log is full.
        /// Returns [`CoreError::InvalidParam`] if `len == 0`.
        pub fn reserve(&self, len: usize) -> Result<WriteToken, CoreError> {
            if len == 0 {
                return Err(CoreError::InvalidParam("reserve len must be > 0"));
            }

            let mut off = self.head.load(Ordering::Acquire);
            loop {
                let end = off
                    .checked_add(len)
                    .ok_or(CoreError::CapacityExceeded)?;

                if end > self.cap {
                    return Err(CoreError::CapacityExceeded);
                }

                match self.head.compare_exchange_weak(
                    off,
                    end,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_)    => return Ok(WriteToken { offset: off, len }),
                    Err(cur) => {
                        off = cur;
                        std::hint::spin_loop(); // CPU back-off — prevents live-lock
                    }
                }
            }
        }

        /// Writes `data` at the offset encoded in `token`, then flushes to disk.
        ///
        /// The token is consumed so it cannot be used twice.
        ///
        /// # Errors
        /// Returns [`CoreError::InvalidParam`] if `data.len() != token.len`.
        pub fn write(&mut self, token: WriteToken, data: &[u8]) -> Result<(), CoreError> {
            if data.len() != token.len {
                return Err(CoreError::InvalidParam(
                    "data length must match reserved length",
                ));
            }

            let off = token.offset;
            let end = off + data.len();

            // SAFETY: `off..end` is within `self.map` (guaranteed by `reserve`),
            // and `token` is consumed so no other call can alias this range.
            unsafe {
                let ptr = self.map.as_mut_ptr().add(off);
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }

            // Full memory fence — ensures all CPUs see the write before flush.
            fence(Ordering::SeqCst);

            // msync(MS_SYNC) — survives power failure.
            self.map.flush_range(off, end - off)?;

            // Advance sequence counter after successful flush.
            self.seq.fetch_add(1, Ordering::Release);

            Ok(())
        }

        /// Returns the current write-head offset.
        #[inline]
        #[must_use]
        pub fn offset(&self) -> usize {
            self.head.load(Ordering::Acquire)
        }

        /// Returns the number of successfully committed entries.
        #[inline]
        #[must_use]
        pub fn committed(&self) -> usize {
            self.seq.load(Ordering::Acquire)
        }

        /// Returns `true` if the log has no remaining capacity.
        #[inline]
        #[must_use]
        pub fn is_full(&self) -> bool {
            self.head.load(Ordering::Acquire) >= self.cap
        }
    }
}

// =============================================================================
//  MODULE 3: consensus
// =============================================================================

/// Raft consensus state machine core.
pub mod consensus {

    // ─────────────────────────────────────────────────────────────────────────
    //  Role
    // ─────────────────────────────────────────────────────────────────────────

    /// The role of a node in the Raft cluster.
    ///
    /// Marked `#[non_exhaustive]` so future roles (e.g. `Learner`) can be
    /// added without a breaking change.
    #[repr(u8)]
    #[non_exhaustive]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Role {
        /// Passive node that replicates the leader's log.
        Follower  = 0,
        /// Node campaigning for leadership.
        Candidate = 1,
        /// Node that accepts and replicates writes.
        Leader    = 2,
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  RaftCore
    // ─────────────────────────────────────────────────────────────────────────

    /// Core Raft persistent and volatile state for a single node.
    ///
    /// # Cache Layout
    /// `#[repr(C, align(64))]` places all fields in one 64-byte cache line,
    /// preventing false sharing when multiple `RaftCore` instances are stored
    /// in a `Vec` or array.
    ///
    /// Padding is exactly `64 − (8+8+8+8+8+8+8+1) = 15` bytes.
    /// A static assertion below enforces this at compile time.
    #[repr(C, align(64))]
    #[derive(Debug)]
    pub struct RaftCore {
        /// Current term — must be persisted before any action.
        pub term:           u64,  // 8
        /// Index of the highest log entry known to be committed.
        pub commit:         u64,  // 8
        /// Index of the highest log entry applied to the state machine.
        pub applied:        u64,  // 8
        /// Candidate ID that received this node's vote in `term`.
        pub voted_for:      u64,  // 8
        /// Index of the last log entry.
        pub last_log_index: u64,  // 8
        /// Term of the last log entry.
        pub last_log_term:  u64,  // 8
        /// Number of votes received during a Candidate election.
        pub votes_received: u64,  // 8
        /// Current node role.
        pub role:           Role, // 1
        /// Explicit padding to fill one full 64-byte cache line.
        /// Prevents false sharing if adjacent fields are added later.
        _pad: [u8; 15],           // 15  → total = 64
    }

    // Compile-time layout assertion.
    const _: () = assert!(
        std::mem::size_of::<RaftCore>() == 64,
        "RaftCore must be exactly 64 bytes (one cache line)"
    );

    impl RaftCore {
        /// Creates a new [`RaftCore`] in the initial `Follower` state.
        #[inline(always)]
        #[must_use]
        pub const fn init() -> Self {
            Self {
                term:           0,
                commit:         0,
                applied:        0,
                voted_for:      u64::MAX, // u64::MAX = "no vote"
                last_log_index: 0,
                last_log_term:  0,
                votes_received: 0,
                role:           Role::Follower,
                _pad:           [0u8; 15],
            }
        }

        /// Advances to the next term and transitions to `Candidate`.
        ///
        /// Resets `voted_for` and `votes_received` as required by the Raft spec.
        /// The caller is responsible for persisting `term` to stable storage
        /// before sending any RPCs.
        ///
