use super::gridbuffer::GridBuffer;
use crate::sniper::SimpleFeatures;

/// SimpleFeatures iterator to GridBuffer iterator.
///
/// return iterator for the result.
pub struct SimpleFeaturesBatcher<I: Iterator<Item = SimpleFeatures>> {
    features: I,
    buffer: GridBuffer,
    row: usize,
    num_rows: usize,
}

impl<I: Iterator<Item = SimpleFeatures>> SimpleFeaturesBatcher<I> {
    pub fn new(features: I, num_rows: usize) -> Self {
        Self {
            features,
            buffer: GridBuffer::new(),
            row: 0,
            num_rows,
        }
    }
}

impl<I: Iterator<Item = SimpleFeatures>> Iterator for SimpleFeaturesBatcher<I> {
    type Item = GridBuffer;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(features) = self.features.next() {
            if !self.buffer.is_initialized() {
                let num_cols = features.sparse_feature.len() + features.dense_feature.len();

                let cols = (0..num_cols).map(|x| x as u32).collect::<Vec<_>>();
                self.buffer.init(self.num_rows, cols);
            }

            let mut col = 0;
            for sparse_feature in features.sparse_feature.iter() {
                // Remove prefix.
                let u64_values = &sparse_feature
                    .values
                    .iter()
                    .map(|x| *x & (1_u64 << 52 - 1))
                    .collect::<Vec<_>>();

                self.buffer.push_u64_values(self.row, col, &u64_values);

                col += 1;
            }

            for dense_feature in features.dense_feature.iter() {
                let f32_values = &dense_feature.values;
                self.buffer.push_f32_values(self.row, col, f32_values);

                col += 1;
            }

            self.row += 1;

            if self.row == self.num_rows {
                let gridbuffer = self.buffer.clone();
                self.buffer.clear();

                self.row = 0;
                return Some(gridbuffer);
            }
        }

        None
    }
}
