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
        loop {
            let offset = self.cursor.load(Ordering::Acquire);
            if offset + size > self.capacity {
                return None;
            }
            if self
                .cursor
                .compare_exchange_weak(offset, offset + size, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(&self.slab[offset..offset + size]);
            }
        }
    }
}

unsafe impl Send for TensorArena {}
unsafe impl Sync for TensorArena {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_reset() {
        let arena = TensorArena::new(1024);
        let a = arena.alloc(512);
        assert!(a.is_some());
        assert_eq!(a.unwrap().len(), 512);
        arena.reset();
        assert_eq!(arena.cursor.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_overflow_returns_none() {
        let arena = TensorArena::new(10);
        let a = arena.alloc(20);
        assert!(a.is_none());
    }

    #[test]
    fn test_sequential_alloc() {
        let arena = TensorArena::new(100);
        let a = arena.alloc(50);
        assert!(a.is_some());
        let b = arena.alloc(50);
        assert!(b.is_some());
        let c = arena.alloc(1);
        assert!(c.is_none());
    }
}
