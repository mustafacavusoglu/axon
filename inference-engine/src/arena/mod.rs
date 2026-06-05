use std::sync::atomic::{AtomicUsize, Ordering};

pub struct TensorArena {
    slab: Vec<u8>,
    cursor: AtomicUsize,
    capacity: usize,
}

impl TensorArena {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            slab: vec![0u8; cap],
            cursor: AtomicUsize::new(0),
            capacity: cap,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn reset(&self) {
        self.cursor.store(0, Ordering::Release);
    }

    pub fn alloc(&self, size: usize) -> Option<&[u8]> {
        let offset = self.cursor.fetch_add(size, Ordering::SeqCst);
        if offset + size > self.capacity {
            self.cursor.store(0, Ordering::Release);
            return None;
        }
        Some(&self.slab[offset..offset + size])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_reset() {
        let arena = TensorArena::new(1024);
        let a = arena.alloc(512);
        assert!(a.is_some());
        arena.reset();
        assert_eq!(arena.cursor.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_overflow_returns_none() {
        let arena = TensorArena::new(10);
        let a = arena.alloc(20);
        assert!(a.is_none());
    }
}
