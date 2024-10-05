# 性能

测试压缩与读写性能。

输入数据为 `SimpleFeatures` 格式序列化后 `base64` 编码的文本文件。其中有 `76` 个 `sparse` 特征，`5` 个 `dense` 特征。

文件路径: `resources/simple_features_nohash_96.txt`。

## 性能对比


| 格式                     | 压缩方法              | 文件大小  | 压缩率   | 读取耗时   | 读取耗时提升 |
|-------------------------|----------------------|----------|---------|-----------|-------------|
| SimpleFeatures (proto)  | 无                   | 378K     | 100%    | 22000 us  | 1x          |
| GridBuffer              | BitPacker4x          | 250K     | 66.1%   | 6000 us   | 3.67x       |
| GridBuffer              | BitPacker4x (sorted) | 436K     | 115%    | -         | -           |
| GridBuffer              | BitPacker8x          | 250K     | 66.1%   | 6000 us   | 3.67x       |

可以看出：

1. GridBuffer 使用 BitPacker4x 和 BitPacker8x 压缩效果最好，都能将文件大小压缩到原来的 66.1%。
2. 使用排序后的 BitPacker4x 压缩反而增加了文件大小，达到原始大小的 115%，可能是因为需要额外存储索引信息。
3. BitPacker4x 和 BitPacker8x 的压缩效果相同，这可能表明在当前数据集上，这两种方法没有明显区别。
4. 在读取性能方面，GridBuffer 格式（无论是使用 BitPacker4x 还是 BitPacker8x）比 SimpleFeatures 格式快约 3.67 倍，耗时从 22000 us 减少到 6000 us。
5. GridBuffer 格式不仅在大多数情况下实现了更好的压缩率，还显著提高了读取速度，展现了其在数据存储和访问方面的优势。


## 评测详情

### 压缩

#### `SimpleFeatures` `proto` 数据

原始数据为 `96` 行，每行表示一条样本。文件大小为: `378K`。

#### `GridBuffer` 格式

##### `BitPacker4x` 压缩

按 `batch_size` 为 `16` 进行压缩。

结果文件为: `resources/gridbuffers_nohash_row_16_col_81_bitpacking4x.txt`

结果为 `6` 行，大小为: `250K`。压缩率为 `66.1%`。

##### `BitPacker4x` `sorted` 压缩

即先将数据组装成 `GridBuffer` 格式，然后对 `u64_data` 进行排序，之后再用 `BitPacker4x` 压缩。

结果为

结果文件为: `resources/gridbuffers_nohash_row_16_col_81_sorted.txt`

结果为 `6` 行，大小为: `436K`。压缩率为 `115%`。比压缩之前还要大。可能是因为需要额外保存下标。

##### `BitPacker8x` 压缩

按 `batch_size` 为 `16` 进行压缩。

结果文件为: `resources/gridbuffers_nohash_row_16_col_81_bitpacking8x.txt`

结果为 `6` 行，大小为: `250K`。压缩率为 `66.1%`。

与 `BitPacker4x` 结果一样。

### 读取

#### `SimpleFeatures` `proto` 数据

单测函数: `read_simple_features_from_file`

耗时: `22000 us`

#### `GridBuffer` 格式

##### `BitPacker4x` 压缩

单测函数: `timing_read_gridbuffer_from_file_with_bitpacking4x`

耗时: `6000 us`

相比 `SimpleFeatures` `proto` 数据，`GridBuffer` 格式压缩后读取速度快了 `3.7` 倍。

##### `BitPacker4x` `sorted` 压缩

##### `BitPacker8x` 压缩

单测函数: `timing_read_gridbuffer_from_file_with_bitpacking8x`

耗时: `6000 us`

相比 `SimpleFeatures` `proto` 数据，`GridBuffer` 格式压缩后读取速度快了 `3.7` 倍。

与 `BitPacker4x` 结果一样。

### 写入