#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![deny(missing_docs, unsafe_op_in_unsafe_fn)]
#![forbid(clippy::unwrap_used, clippy::expect_used)]

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("size exceeds limit: {0} bytes")]
    SizeLimit(usize),
    #[error("memtable capacity exceeded")]
    CapacityExceeded,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wal corruption: expected crc={expected:#010x}, got={actual:#010x}")]
    WalCorruption { expected: u32, actual: u32 },
    #[error("empty payload")]
    EmptyPayload,
    #[error("wal capacity exceeded")]
    WalFull,
    #[error("router queue full on shard {shard}")]
    RouterFull { shard: usize },
}

pub mod bloom {
    use std::sync::{atomic::{fence, AtomicU8, Ordering}, Arc};
    use super::EngineError;

    const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME:        u64 = 0x0000_0100_0000_01b3;

    #[repr(align(64))]
    #[derive(Debug)]
    pub struct BloomFilter {
        raw:  Arc<[AtomicU8]>,
        k:    u8,
        mask: usize,
    }

    unsafe impl Send for BloomFilter {}
    unsafe impl Sync for BloomFilter {}

    impl BloomFilter {
        pub fn new(n: usize, p: f64) -> Result<Self, EngineError> {
            if n == 0 || !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err(EngineError::SizeLimit(0));
            }
            let m_bits = (-(n as f64) * p.ln() / std::f64::consts::LN_2.powi(2)).ceil() as u64;
            let m_p2 = m_bits
                .checked_next_power_of_two()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(EngineError::SizeLimit(usize::MAX))?;
            let byte_count = m_p2 >> 3;
            if byte_count == 0 { return Err(EngineError::SizeLimit(0)); }
            let k = ((m_p2 as f64 / n as f64) * std::f64::consts::LN_2).round().clamp(1.0, 31.0) as u8;
            let raw: Arc<[AtomicU8]> = (0..byte_count).map(|_| AtomicU8::new(0)).collect();
            Ok(Self { raw, k, mask: m_p2 - 1 })
        }

        #[inline]
        pub fn insert(&self, h1: u64, h2: u64) {
            if h2 == 0 { return; }
            for i in 0..self.k {
                let bit  = self.bit_index(h1, h2, i);
                let byte = bit >> 3;
                // SAFETY: bit_index guarantees byte < raw.len()
                self.raw[byte].fetch_or(1u8 << (bit & 7), Ordering::Release);
            }
            fence(Ordering::SeqCst);
        }

        #[inline]
        #[must_use]
        pub fn contains(&self, h1: u64, h2: u64) -> bool {
            if h2 == 0 { return false; }
            fence(Ordering::Acquire);
            let mut present = true;
            for i in 0..self.k {
                let bit  = self.bit_index(h1, h2, i);
                let byte = bit >> 3;
                present &= (self.raw[byte].load(Ordering::Relaxed) >> (bit & 7)) & 1 == 1;
            }
            present
        }

        pub fn clear(&self) {
            for byte in self.raw.iter() { byte.store(0, Ordering::Release); }
            fence(Ordering::SeqCst);
        }

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

        #[inline(always)]
        #[must_use]
        pub fn fnv1a(data: &[u8]) -> u64 {
            data.iter().fold(FNV_OFFSET_BASIS, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
        }

        #[inline(always)]
        #[must_use]
        pub fn fnv1a_seeded(data: &[u8], seed: u64) -> u64 {
            data.iter().fold(seed, |h, &b| (h ^ b as u64).wrapping_mul(FNV_PRIME))
        }
    }
}

pub mod memtable {
    use crossbeam_skiplist::SkipMap;
    use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};
    use super::EngineError;

    #[derive(Debug)]
    pub struct MemTable {
        idx:       Arc<SkipMap<Arc<[u8]>, Arc<[u8]>>>,
        usage:     AtomicUsize,
        max_bytes: usize,
    }

    impl MemTable {
        pub fn new(max_bytes: usize) -> Result<Self, EngineError> {
            if max_bytes == 0 { return Err(EngineError::SizeLimit(0)); }
            Ok(Self { idx: Arc::new(SkipMap::new()), usage: AtomicUsize::new(0), max_bytes })
        }

        #[must_use = "check for CapacityExceeded"]
        pub fn put(&self, k: &[u8], v: &[u8]) -> Result<(), EngineError> {
            let cost = k.len() + v.len() + 176;
            let prev = self.usage.fetch_add(cost, Ordering::AcqRel);
            if prev + cost > self.max_bytes {
                self.usage.fetch_sub(cost, Ordering::AcqRel);
                return Err(EngineError::CapacityExceeded);
            }
            self.idx.insert(Arc::from(k), Arc::from(v));
            Ok(())
        }

        #[must_use]
        pub fn get(&self, k: &[u8]) -> Option<Arc<[u8]>> {
            self.idx.get(k).map(|e| Arc::clone(e.value()))
        }

        pub fn delete(&self, k: &[u8]) {
            if let Some(e) = self.idx.get(k) {
                let freed = e.key().len() + e.value().len() + 176;
                drop(e);
                self.idx.remove(k);
                self.usage.fetch_sub(freed, Ordering::AcqRel);
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = (Arc<[u8]>, Arc<[u8]>)> + '_ {
            self.idx.iter().map(|e| (Arc::clone(e.key()), Arc::clone(e.value())))
        }

        #[inline(always)] #[must_use]
        pub fn footprint(&self) -> usize { self.usage.load(Ordering::Acquire) }

        #[inline(always)] #[must_use]
        pub fn is_empty(&self) -> bool { self.idx.is_empty() }
    }

    impl Default for MemTable {
        fn default() -> Self {
            // SAFETY: 64 MiB > 0 is trivially true
            Self::new(64 * 1024 * 1024).unwrap_or_else(|_| unreachable!())
        }
    }
      }
pub mod wal {
    use memmap2::MmapMut;
    use std::{fs::OpenOptions, path::Path, sync::atomic::{fence, AtomicUsize, Ordering}};
    use super::EngineError;

    #[derive(Debug)]
    #[must_use]
    pub struct WriteToken { offset: usize, len: usize }

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
                .field("cap",       &self.cap)
                .field("committed", &self.seq.load(Ordering::Relaxed))
                .finish()
        }
    }

    impl Wal {
        pub fn open<P: AsRef<Path>>(p: P, sz: usize) -> Result<Self, EngineError> {
            if sz == 0 { return Err(EngineError::SizeLimit(0)); }
            if sz > i64::MAX as usize { return Err(EngineError::SizeLimit(sz)); }
            let fd = OpenOptions::new().read(true).write(true).create(true).open(p.as_ref())?;
            fd.set_len(sz as u64)?;
            // SAFETY: fd is valid, open for read+write, sz > 0 and sz <= i64::MAX.
            // The mmap is exclusively owned by Self.
            let mmap = unsafe { MmapMut::map_mut(&fd)? };
            Ok(Self { map: mmap, head: AtomicUsize::new(0), cap: sz, seq: AtomicUsize::new(0) })
        }

        pub fn reserve(&self, len: usize) -> Result<WriteToken, EngineError> {
            if len == 0 { return Err(EngineError::EmptyPayload); }
            let mut off = self.head.load(Ordering::Acquire);
            loop {
                let end = off.checked_add(len).ok_or(EngineError::WalFull)?;
                if end > self.cap { return Err(EngineError::WalFull); }
                match self.head.compare_exchange_weak(off, end, Ordering::AcqRel, Ordering::Acquire) {
                    Ok(_)    => return Ok(WriteToken { offset: off, len }),
                    Err(cur) => { off = cur; std::hint::spin_loop(); }
                }
            }
        }

        pub fn write(&mut self, token: WriteToken, data: &[u8]) -> Result<(), EngineError> {
            if data.len() != token.len { return Err(EngineError::SizeLimit(data.len())); }
            let off = token.offset;
            // SAFETY: off..off+data.len() is within self.map, guaranteed by reserve().
            // Token is consumed so no other call aliases this range.
            // &mut self ensures exclusive mutable access to self.map.
            unsafe {
                let dst = self.map.as_mut_ptr().add(off);
                std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
            }
            fence(Ordering::SeqCst);
            self.map.flush_range(off, data.len())?;
            self.seq.fetch_add(1, Ordering::Release);
            Ok(())
        }

        #[inline(always)] #[must_use]
        pub fn offset(&self) -> usize { self.head.load(Ordering::Acquire) }

        #[inline(always)] #[must_use]
        pub fn committed(&self) -> usize { self.seq.load(Ordering::Acquire) }

        #[inline(always)] #[must_use]
        pub fn is_full(&self) -> bool { self.head.load(Ordering::Acquire) >= self.cap }
    }
}

pub mod sstable {
    use std::sync::Arc;
    use super::{bloom::BloomFilter, EngineError};

    const BLOOM_SEED_2: u64 = 0xdead_beef_cafe_babe;

    #[derive(Debug, Clone)]
    pub struct SsEntry {
        pub key:   Arc<[u8]>,
        pub value: Arc<[u8]>,
        pub seq:   u64,
    }

    #[repr(align(64))]
    pub struct SsTable {
        entries: Arc<[SsEntry]>,
        bloom:   BloomFilter,
    }

    impl std::fmt::Debug for SsTable {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SsTable").field("len", &self.entries.len()).finish()
        }
    }

    impl SsTable {
        pub fn from_sorted(entries: Vec<SsEntry>) -> Result<Self, EngineError> {
            if entries.is_empty() { return Err(EngineError::EmptyPayload); }
            for w in entries.windows(2) {
                if w[0].key > w[1].key { return Err(EngineError::SizeLimit(0)); }
            }
            let bloom = BloomFilter::new(entries.len(), 0.01)?;
            for e in &entries {
                let h1 = BloomFilter::fnv1a(&e.key);
                let h2 = BloomFilter::fnv1a_seeded(&e.key, BLOOM_SEED_2);
                bloom.insert(h1, h2);
            }
            Ok(Self { entries: entries.into(), bloom })
        }

        #[must_use]
        pub fn get(&self, key: &[u8]) -> Option<Arc<[u8]>> {
            let h1 = BloomFilter::fnv1a(key);
            let h2 = BloomFilter::fnv1a_seeded(key, BLOOM_SEED_2);
            if !self.bloom.contains(h1, h2) { return None; }
            self.entries
                .binary_search_by(|e| e.key.as_ref().cmp(key))
                .ok()
                .map(|i| Arc::clone(&self.entries[i].value))
        }

        #[inline(always)] #[must_use]
        pub fn len(&self) -> usize { self.entries.len() }

        #[inline(always)] #[must_use]
        pub fn is_empty(&self) -> bool { self.entries.is_empty() }
    }
}

pub mod consensus {
    #[repr(u8)]
    #[non_exhaustive]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Role { Follower = 0, Candidate = 1, Leader = 2 }

    #[repr(C, align(64))]
    #[derive(Debug)]
    pub struct RaftCore {
        pub term:           u64,
        pub commit:         u64,
        pub applied:        u64,
        pub voted_for:      u64,
        pub last_log_index: u64,
        pub last_log_term:  u64,
        pub votes_received: u64,
        pub role:           Role,
        _pad: [u8; 7],
    }

    const _: () = assert!(std::mem::size_of::<RaftCore>() == 64);

    impl RaftCore {
        #[inline(always)] #[must_use]
        pub const fn init() -> Self {
            Self {
                term: 0, commit: 0, applied: 0, voted_for: u64::MAX,
                last_log_index: 0, last_log_term: 0, votes_received: 0,
                role: Role::Follower, _pad: [0u8; 7],
            }
        }

        pub fn next_term(&mut self) -> Result<(), ()> {
            if self.role == Role::Leader { return Err(()); }
            self.term           = self.term.checked_add(1).ok_or(())?;
            self.role           = Role::Candidate;
            self.voted_for      = u64::MAX;
            self.votes_received = 0;
            Ok(())
        }

        pub fn record_vote(&mut self, cluster_size: usize) -> Result<bool, ()> {
            if self.role != Role::Candidate { return Err(()); }
            self.votes_received = self.votes_received.saturating_add(1);
            Ok(self.votes_received >= (cluster_size / 2 + 1) as u64)
        }

        pub fn become_leader(&mut self) -> Result<(), ()> {
            if self.role != Role::Candidate { return Err(()); }
            self.role           = Role::Leader;
            self.votes_received = 0;
            Ok(())
        }

        pub fn step_down(&mut self, new_term: u64) {
            if new_term > self.term { self.term = new_term; self.voted_for = u64::MAX; }
            self.role = Role::Follower;
        }

        #[inline(always)] #[must_use]
        pub fn has_pending_apply(&self) -> bool { self.commit > self.applied }
    }
}

pub mod router {
    use crossbeam::channel::{bounded, Receiver, Sender, TrySendError};
    use std::sync::Arc;
    use super::EngineError;

    #[derive(Debug)]
    pub struct ShardRouter {
        senders:    Arc<[Sender<Box<[u8]>>]>,
        _receivers: Vec<Receiver<Box<[u8]>>>,
        mask:       usize,
        capacity:   usize,
    }

    impl ShardRouter {
        pub fn new(shard_count: usize, queue_capacity: usize) -> Result<Self, EngineError> {
            if shard_count == 0 || queue_capacity == 0 { return Err(EngineError::SizeLimit(0)); }
            let n = shard_count.next_power_of_two().min(16);
            let mut senders   = Vec::with_capacity(n);
            let mut receivers = Vec::with_capacity(n);
            for _ in 0..n { let (tx, rx) = bounded(queue_capacity); senders.push(tx); receivers.push(rx); }
            Ok(Self { senders: senders.into(), _receivers: receivers, mask: n - 1, capacity: queue_capacity })
        }

        #[inline(always)]
        pub fn dispatch(&self, h: u64, payload: Box<[u8]>) -> Result<(), EngineError> {
            if payload.is_empty() { return Err(EngineError::EmptyPayload); }
            let shard = (h as usize) & self.mask;
            self.senders[shard].try_send(payload).map_err(|e| match e {
                TrySendError::Full(_) | TrySendError::Disconnected(_) =>
                    EngineError::RouterFull { shard },
            })
        }

        #[must_use]
        pub fn receiver(&self, i: usize) -> Option<&Receiver<Box<[u8]>>> { self._receivers.get(i) }

        #[inline(always)] #[must_use]
        pub fn shard_count(&self) -> usize { self.mask + 1 }

        #[inline(always)] #[must_use]
        pub fn queue_capacity(&self) -> usize { self.capacity }
    }
              }
#[cfg(test)]
mod tests {
    use super::*;

    mod bloom_tests {
        use super::bloom::BloomFilter;

        #[test]
        fn no_false_negatives() {
            let bf = BloomFilter::new(1_000, 0.01).unwrap();
            for i in 0u64..500 { bf.insert(i, i.wrapping_add(1)); }
            for i in 0u64..500 { assert!(bf.contains(i, i.wrapping_add(1))); }
        }

        #[test]
        fn clear_resets_all_bits() {
            let bf = BloomFilter::new(100, 0.01).unwrap();
            bf.insert(1, 2);
            bf.clear();
            assert!(!bf.contains(1, 2));
        }

        #[test]
        fn degenerate_h2_zero_is_safe() {
            let bf = BloomFilter::new(100, 0.01).unwrap();
            bf.insert(42, 0);
            assert!(!bf.contains(42, 0));
        }

        #[test]
        fn empty_slice_hash_is_deterministic() {
            assert_eq!(BloomFilter::fnv1a(b""), BloomFilter::fnv1a(b""));
        }

        #[test]
        fn fpr_stays_bounded() {
            let bf = BloomFilter::new(1_000, 0.01).unwrap();
            assert!(bf.current_fpr(1_000) <= 0.05);
        }
    }

    mod memtable_tests {
        use super::memtable::MemTable;
        use super::EngineError;

        #[test]
        fn put_get_roundtrip() {
            let m = MemTable::new(1024 * 1024).unwrap();
            m.put(b"hello", b"world").unwrap();
            assert_eq!(m.get(b"hello").unwrap().as_ref(), b"world");
        }

        #[test]
        fn delete_removes_entry() {
            let m = MemTable::new(1024 * 1024).unwrap();
            m.put(b"k", b"v").unwrap();
            m.delete(b"k");
            assert!(m.get(b"k").is_none());
        }

        #[test]
        fn capacity_exceeded_returns_error() {
            let m = MemTable::new(1).unwrap();
            assert!(matches!(m.put(b"k", b"v"), Err(EngineError::CapacityExceeded)));
        }

        #[test]
        fn is_empty_on_new() {
            assert!(MemTable::default().is_empty());
        }
    }

    mod wal_tests {
        use super::wal::Wal;
        use super::EngineError;

        #[test]
        fn reserve_and_write_roundtrip() {
            let dir  = tempfile::tempdir().unwrap();
            let path = dir.path().join("test.wal");
            let mut wal = Wal::open(&path, 4096).unwrap();
            let tok     = wal.reserve(5).unwrap();
            wal.write(tok, b"hello").unwrap();
            assert_eq!(wal.committed(), 1);
        }

        #[test]
        fn overflow_returns_wal_full() {
            let dir  = tempfile::tempdir().unwrap();
            let path = dir.path().join("full.wal");
            let wal  = Wal::open(&path, 8).unwrap();
            let _t   = wal.reserve(8).unwrap();
            assert!(matches!(wal.reserve(1), Err(EngineError::WalFull)));
        }

        #[test]
        fn zero_reserve_returns_empty_payload() {
            let dir  = tempfile::tempdir().unwrap();
            let path = dir.path().join("zero.wal");
            let wal  = Wal::open(&path, 4096).unwrap();
            assert!(matches!(wal.reserve(0), Err(EngineError::EmptyPayload)));
        }
    }

    mod raft_tests {
        use super::consensus::{RaftCore, Role};

        #[test]
        fn exactly_one_cache_line() {
            assert_eq!(std::mem::size_of::<RaftCore>(), 64);
        }

        #[test]
        fn quorum_detection() {
            let mut c = RaftCore::init();
            c.next_term().unwrap();
            assert!(!c.record_vote(5).unwrap());
            assert!(!c.record_vote(5).unwrap());
            assert!(c.record_vote(5).unwrap());
        }

        #[test]
        fn leader_cannot_start_election() {
            let mut c = RaftCore::init();
            c.next_term().unwrap();
            c.become_leader().unwrap();
            assert!(c.next_term().is_err());
        }

        #[test]
        fn step_down_resets_state() {
            let mut c = RaftCore::init();
            c.next_term().unwrap();
            c.step_down(99);
            assert_eq!(c.term,      99);
            assert_eq!(c.voted_for, u64::MAX);
            assert_eq!(c.role,      Role::Follower);
        }

        #[test]
        fn become_leader_clears_votes() {
            let mut c = RaftCore::init();
            c.next_term().unwrap();
            c.record_vote(3).unwrap();
            c.become_leader().unwrap();
            assert_eq!(c.votes_received, 0);
        }
    }

    mod sstable_tests {
        use std::sync::Arc;
        use super::sstable::{SsEntry, SsTable};
        use super::EngineError;

        fn entry(k: &[u8], v: &[u8]) -> SsEntry {
            SsEntry { key: Arc::from(k), value: Arc::from(v), seq: 0 }
        }

        #[test]
        fn get_existing_key() {
            let t = SsTable::from_sorted(vec![
                entry(b"a", b"1"),
                entry(b"b", b"2"),
            ]).unwrap();
            assert_eq!(t.get(b"a").unwrap().as_ref(), b"1");
        }

        #[test]
        fn get_missing_key_returns_none() {
            let t = SsTable::from_sorted(vec![entry(b"a", b"1")]).unwrap();
            assert!(t.get(b"z").is_none());
        }

        #[test]
        fn unsorted_input_rejected() {
            let result = SsTable::from_sorted(vec![
                entry(b"b", b"2"),
                entry(b"a", b"1"),
            ]);
            assert!(matches!(result, Err(EngineError::SizeLimit(_))));
        }

        #[test]
        fn empty_input_rejected() {
            assert!(matches!(
                SsTable::from_sorted(vec![]),
                Err(EngineError::EmptyPayload)
            ));
        }
    }

    mod router_tests {
        use super::router::ShardRouter;
        use super::EngineError;

        #[test]
        fn dispatch_and_receive() {
            let r     = ShardRouter::new(4, 128).unwrap();
            let shard = 7usize & (r.shard_count() - 1);
            r.dispatch(7, Box::from(b"ping".as_ref())).unwrap();
            let msg = r.receiver(shard).unwrap().try_recv().unwrap();
            assert_eq!(msg.as_ref(), b"ping");
        }

        #[test]
        fn empty_payload_rejected() {
            let r = ShardRouter::new(4, 128).unwrap();
            assert!(matches!(
                r.dispatch(0, Box::from(b"".as_ref())),
                Err(EngineError::EmptyPayload)
            ));
        }

        #[test]
        fn full_queue_returns_router_full() {
            let r = ShardRouter::new(1, 1).unwrap();
            r.dispatch(0, Box::from(b"a".as_ref())).unwrap();
            assert!(matches!(
                r.dispatch(0, Box::from(b"b".as_ref())),
                Err(EngineError::RouterFull { .. })
            ));
        }

        #[test]
        fn shard_count_is_power_of_two() {
            let r = ShardRouter::new(5, 64).unwrap();
            assert!(r.shard_count().is_power_of_two());
        }
    }
                      }
