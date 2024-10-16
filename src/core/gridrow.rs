use super::gridbuffer::GridBuffer;

/// One row in gridbuffer.
///
/// We use reference and index to represent one row.
pub struct Row<'a> {
    /// The underlying gridbuffer reference.
    pub gridbuffer: &'a GridBuffer,

    /// The index of the row.
    pub row: usize,
}

impl<'a> Row<'a> {
    pub fn new(gridbuffer: &'a GridBuffer, row: usize) -> Self {
        Self { gridbuffer, row }
    }

    pub fn get_u64_values(&self, col: usize) -> &[u64] {
        self.gridbuffer.get_u64_values(self.row, col)
    }

    pub fn get_f32_values(&self, col: usize) -> &[f32] {
        self.gridbuffer.get_f32_values(self.row, col)
    }

    pub fn get_u64(&self, col: usize) -> Option<u64> {
        self.gridbuffer.get_u64(self.row, col)
    }

    pub fn get_f32(&self, col: usize) -> Option<f32> {
        self.gridbuffer.get_f32(self.row, col)
    }
}
