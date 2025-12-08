namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        public Tensor(float[] data, int[] size) : this(data, size, Util.Strides.CStrides(size))
        {
        }
        /// <summary>
        /// メモリ上で連続なデータを持つ新しいテンソルを返す.
        /// 既に連続している場合は自分自身を返す.
        /// </summary>
        public static Tensor Contiguous(Tensor tensor) {
            if (tensor.IsContiguous) return tensor;
            
            // データを実体化して新しいTensorを作成
            return new Tensor(
                tensor.GetContiguousData(),
                tensor.Size
            );
        }

        /// <summary>
        /// ストライドを考慮して、メモリ上で連続した1次元配列データを生成して返す.
        /// </summary>
        public float[] GetContiguousData() {
            var totalSize = _totalSize;
            var outData = new float[totalSize];

            int ndim = _ndim;
            int[] counters = new int[ndim];

            var srcStrides = _strides; 
            var srcData = _data;
            var size = _size;

            for (uint i = 0; i < totalSize; i++)
            {
                int srcIndex = 0;
                for (int d = 0; d < ndim; d++)
                {
                    srcIndex += counters[d] * srcStrides[d];
                }

                outData[i] = srcData[srcIndex];

                for (int d = ndim - 1; d >= 0; d--)
                {
                    counters[d]++;
                    if (counters[d] < size[d])
                    {
                        break;
                    }
                    counters[d] = 0;
                }
            }
            return outData;
        }
    }
}