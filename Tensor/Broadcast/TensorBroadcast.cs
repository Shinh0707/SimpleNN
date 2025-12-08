namespace SimpleNN.Tensor
{
    using System;
    public partial class Tensor
    {
        public static bool CheckBroadcast(int size, int targetSize, bool invokeException=false)
        {
            if (size == targetSize)
            {
                return false;
            }
            if (size != 1)
            {
                if (!invokeException) return false;
                throw new ArgumentException($"size {size}, but must be 1 to broadcast to {targetSize}.");
            }
            return true;
        }
        public static bool TryBroadcast(int size, int stride, int targetSize, out int newSize, out int newStride)
        {
            if (size == targetSize)
            {
                newSize = size;
                newStride = stride;
                return false;
            }
            if (size != 1)
            {
                throw new ArgumentException($"size {size}, but must be 1 to broadcast to {targetSize}.");
            }
            newSize = targetSize;
            newStride = 0;
            return true;
        }
        // <summary>
        /// 指定された次元をブロードキャスト（拡張）した新しいTensor(View)を返す.
        /// データはコピーせず, ストライドを0にすることで論理的に拡張する.
        /// </summary>
        /// <param name="dim">拡張する次元</param>
        /// <param name="targetSize">目標のサイズ</param>
        /// <returns>ブロードキャストされたTensor</returns>
        public static Tensor Broadcast(Tensor tensor, int dim, int targetSize)
        {
            try
            {
                if (CheckBroadcast(tensor._size[dim], targetSize, true))
                {
                    var newShape = (int[])tensor.Size.Clone();
                    var newStrides = (int[])tensor.Strides.Clone();
                    newShape[dim] = targetSize;
                    newStrides[dim] = 0;
                    return new Tensor(tensor._data, newShape, newStrides);
                }
            }
            catch(ArgumentException e)
            {
                throw new ArgumentException($"At Dimension {dim}, {e.Message}");
            }
            return tensor;
        }
        
        public void Broadcast(int dim, int targetSize)
        {
            try
            {
                if (CheckBroadcast(_size[dim], targetSize, true))
                {
                    Size[dim] = targetSize;
                    Strides[dim] = 0;
                    Refresh();
                }
            }
            catch(ArgumentException e)
            {
                throw new ArgumentException($"At Dimension {dim}, {e.Message}");
            }
        }
        public static Tensor Broadcast(Tensor tensor, int[] targetSize) {
            if (tensor.IsScaler)
            {
                return Fill(targetSize.Length > 0 ? targetSize : new int[]{1}, tensor.Item());
            }
            if (tensor.NDim != targetSize.Length)
            {
                return Reshape(tensor, targetSize);
            }
            
            int targetRank = targetSize.Length;
            var newSize = new int[targetRank];
            var newStrides = new int[targetRank];

            int offset = targetRank - tensor.NDim;

            for (int i = 0; i < targetRank; ++i)
            {
                int tDim = targetSize[i];
                newSize[i] = tDim;

                int srcIndex = i - offset;

                if (srcIndex < 0)
                {
                    newStrides[i] = 0;
                }
                else
                {
                    int sDim = tensor.Size[srcIndex];
                    int sStride = tensor.Strides[srcIndex];

                    if (sDim == tDim)
                    {
                        newStrides[i] = sStride;
                    }
                    else if (sDim == 1)
                    {
                        newStrides[i] = 0;
                    }
                    else
                    {
                        throw new ArgumentException(
                            $"Dimensions match error at dim {i}: {sDim} vs {tDim}.");
                    }
                }
            }
            return new Tensor(tensor._data, newSize, newStrides);
        }
        /// <summary>
        /// 2つのTensorの形状をブロードキャスト可能か検証し, 
        /// 双方を同じ形状に拡張した新しいTensor(View)のペアを返す.
        /// </summary>
        /// <remarks>
        /// ランク（次元数）が異なる場合, 小さい方のTensorの先頭にサイズ1の次元を追加してランクを合わせる.
        /// その後, 各次元のサイズを比較し, 1であれば他方のサイズに合わせて拡張する.
        /// </remarks>
        /// <param name="a">Tensor A</param>
        /// <param name="b">Tensor B</param>
        /// <returns>ブロードキャスト済みのTensor AとBのペア</returns>
        /// <exception cref="ArgumentException">ブロードキャスト不可能な形状の場合にスローされる.</exception>
        private static (Tensor broadA, Tensor broadB) BroadcastBoth(Tensor a, Tensor b)
        {
            // 1. ランク（次元数）を合わせる.
            // ランクが低い方を, 高い方に合わせて先頭にサイズ1の次元を追加（Unsqueeze）する.
            int maxDim = (a.NDim > b.NDim) ? a.NDim : b.NDim;
            
            Tensor tA = null;
            Tensor tB = null;

            if (a.NDim != maxDim)
            {
                tA = PadRankLeft(a, maxDim);
            }

            if (b.NDim != maxDim)
            {
                tB = PadRankLeft(b, maxDim);
            }

            // 2. 各次元のサイズを合わせてブロードキャストする.
            for (int i = 0; i < maxDim; i++)
            {
                int sizeA = (tA ?? a).Size[i];
                int sizeB = (tB ?? b).Size[i];

                if (sizeA == sizeB)
                {
                    continue;
                }

                if (sizeA == 1)
                {
                    if (tA is null)
                    {
                        tA = Broadcast(a, i, sizeB);
                    }
                    else
                    {
                        tA.Broadcast(i, sizeB);
                    }
                }
                else if (sizeB == 1)
                {
                    if (tB is null)
                    {
                        tB = Broadcast(b, i, sizeA);
                    }
                    else
                    {
                        tB.Broadcast(i, sizeA);
                    }
                }
                else
                {
                    throw new ArgumentException(
                        $"Dimensions match error at dim {i}: {sizeA} vs {sizeB}. Both must be equal or 1.");
                }
            }

            return (tA ?? a, tB ?? b);
        }

        /// <summary>
        /// Tensorのランク（次元数）を指定された数になるまで先頭に1を追加して拡張する.
        /// </summary>
        /// <param name="tensor">対象のTensor</param>
        /// <param name="targetRank">目標とするランク</param>
        /// <returns>ランクが拡張されたTensor(View)</returns>
        private static Tensor PadRankLeft(Tensor tensor, int targetRank)
        {
            int currentRank = tensor.NDim;
            if (currentRank >= targetRank)
            {
                return tensor;
            }

            int diff = targetRank - currentRank;
            var newSize = new int[targetRank];
            var newStrides = new int[targetRank];

            // 先頭にサイズ1, ストライド0（拡張用）の次元を埋める.
            for (int i = 0; i < diff; i++)
            {
                newSize[i] = 1;
                newStrides[i] = 0; 
            }

            // 既存のサイズとストライドを後ろにコピーする.
            Array.Copy(tensor.Size, 0, newSize, diff, currentRank);
            Array.Copy(tensor.Strides, 0, newStrides, diff, currentRank);

            // 新しいViewを返す. データの実体は共有される.
            return new Tensor(tensor._data, newSize, newStrides);
        }
    }
}