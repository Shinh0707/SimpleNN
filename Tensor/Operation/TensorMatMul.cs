namespace SimpleNN.Tensor
{
    using System;
    using System.Numerics;
    using System.Runtime.InteropServices;

    public partial class Tensor
    {
        /// <summary>
        /// 2つのテンソルの行列積を計算する.
        /// バッチ次元のブロードキャスト、SIMD(Vector)化、およびキャッシュブロッキングに対応.
        /// </summary>
        /// <param name="a">左側のテンソル (..., M, K)</param>
        /// <param name="b">右側のテンソル (..., K, N)</param>
        /// <returns>計算結果 (..., M, N)</returns>
        public static Tensor MatMul(Tensor a, Tensor b)
        {
            int aDim = a.NDim;
            int bDim = b.NDim;

            if (aDim < 2 || bDim < 2)
            {
                throw new ArgumentException("Tensors must be at least 2D for MatMul.");
            }

            int m = a.Size[aDim - 2];
            int k = a.Size[aDim - 1];
            int k_b = b.Size[bDim - 2];
            int n = b.Size[bDim - 1];

            if (k != k_b)
            {
                throw new ArgumentException($"Shape mismatch: inner dims {k} and {k_b}.");
            }

            // --- ブロードキャスト形状の計算 (既存ロジックと同様) ---
            int batchDimA = aDim - 2;
            int batchDimB = bDim - 2;
            int maxBatchDim = Math.Max(batchDimA, batchDimB);
            int[] batchShape = new int[maxBatchDim];

            for (int i = 0; i < maxBatchDim; i++)
            {
                int offset = i;
                int idxA = batchDimA - 1 - offset;
                int idxB = batchDimB - 1 - offset;
                int sizeA = (idxA >= 0) ? a.Size[idxA] : 1;
                int sizeB = (idxB >= 0) ? b.Size[idxB] : 1;

                if (sizeA != sizeB && sizeA != 1 && sizeB != 1)
                {
                    throw new ArgumentException($"MatMul broadcast error. A:{sizeA}, B:{sizeB}");
                }
                batchShape[maxBatchDim - 1 - i] = Math.Max(sizeA, sizeB);
            }

            // 出力形状
            int[] outShape = new int[maxBatchDim + 2];
            Array.Copy(batchShape, 0, outShape, 0, maxBatchDim);
            outShape[maxBatchDim] = m;
            outShape[maxBatchDim + 1] = n;

            // 結果配列確保
            long totalElementsLong = 1;
            for (int i = 0; i < outShape.Length; i++) totalElementsLong *= outShape[i];
            float[] outData = new float[totalElementsLong];

            // ストライド取得
            int strideAM = a.Strides[aDim - 2];
            int strideAK = a.Strides[aDim - 1];
            int strideBK = b.Strides[bDim - 2];
            int strideBN = b.Strides[bDim - 1];

            float[] dataA = a._data;
            float[] dataB = b._data;

            int batchCount = 1;
            for (int i = 0; i < maxBatchDim; i++) batchCount *= batchShape[i];

            int[] counters = new int[maxBatchDim];
            int offsetOut = 0;
            int matrixSize = m * n;
            bool allowSimd = (strideBN == 1);

            for (int bIdx = 0; bIdx < batchCount; bIdx++)
            {
                int offsetA = 0;
                int offsetB = 0;

                for (int d = 0; d < maxBatchDim; d++)
                {
                    int fromEnd = maxBatchDim - 1 - d;
                    
                    int dimA = (batchDimA - 1) - fromEnd;
                    if (dimA >= 0 && a.Size[dimA] > 1) offsetA += counters[d] * a.Strides[dimA];

                    int dimB = (batchDimB - 1) - fromEnd;
                    if (dimB >= 0 && b.Size[dimB] > 1) offsetB += counters[d] * b.Strides[dimB];
                }
                if (allowSimd)
                {
                    MatMulBlockSimd(m, n, k, dataA, offsetA, strideAM, strideAK, dataB, offsetB, strideBK, outData, offsetOut);
                }
                else
                {
                    MatMulScalar(m, n, k, dataA, offsetA, strideAM, strideAK, dataB, offsetB, strideBK, strideBN, outData, offsetOut);
                }

                offsetOut += matrixSize;

                for (int d = maxBatchDim - 1; d >= 0; d--)
                {
                    counters[d]++;
                    if (counters[d] < batchShape[d]) break;
                    counters[d] = 0;
                }
            }

            return new Tensor(outData, outShape, Util.Strides.CStrides(outShape));
        }

        /// <summary>
        /// キャッシュブロッキングとSIMDを使用した行列積カーネル.
        /// 前提: Bの列方向(N)と出力Cの列方向(N)が連続メモリであること.
        /// </summary>
        private static void MatMulBlockSimd(
            int m, int n, int k,
            float[] A, int offsetA, int strideAM, int strideAK,
            float[] B, int offsetB, int strideBK, // strideBN is assumed 1
            float[] C, int offsetC)
        {
            const int BlockSize = 64; 
            int vecCount = Vector<float>.Count;

            // --- Block Loop (Tiling) ---
            for (int i0 = 0; i0 < m; i0 += BlockSize)
            {
                int iMax = Math.Min(i0 + BlockSize, m);

                for (int k0 = 0; k0 < k; k0 += BlockSize)
                {
                    int kMax = Math.Min(k0 + BlockSize, k);

                    for (int j0 = 0; j0 < n; j0 += BlockSize)
                    {
                        int jMax = Math.Min(j0 + BlockSize, n);
                        for (int i = i0; i < iMax; i++)
                        {
                            int ptrA = offsetA + i * strideAM + k0 * strideAK;
                            int rowCPtrBase = offsetC + i * n;

                            for (int l = k0; l < kMax; l++)
                            {
                                float valA = A[ptrA];
                                ptrA += strideAK;

                                // ゼロスキップ (スパース最適化)
                                if (valA == 0f) continue;

                                Vector<float> vecA = new(valA);
                                int ptrB = offsetB + l * strideBK + j0;
                                int ptrC = rowCPtrBase + j0;

                                int count = jMax - j0;
                                int vecLoops = count / vecCount;
                                ReadOnlySpan<float> spanB = B.AsSpan(ptrB, count);
                                Span<float> spanC = C.AsSpan(ptrC, count);
                                ReadOnlySpan<Vector<float>> vecSpanB = MemoryMarshal.Cast<float, Vector<float>>(spanB);
                                Span<Vector<float>> vecSpanC = MemoryMarshal.Cast<float, Vector<float>>(spanC);
                                for (int v = 0; v < vecLoops; v++)
                                {
                                    // C += A * B
                                    vecSpanC[v] += vecA * vecSpanB[v];
                                }
                                int remainder = vecLoops * vecCount;
                                for (int r = remainder; r < count; r++)
                                {
                                    spanC[r] += valA * spanB[r];
                                }
                            }
                        }
                    }
                }
            }
        }

        private static void MatMulScalar(
            int m, int n, int k,
            float[] A, int offsetA, int strideAM, int strideAK,
            float[] B, int offsetB, int strideBK, int strideBN,
            float[] C, int offsetC)
        {
            for (int i = 0; i < m; i++)
            {
                int baseA = offsetA + i * strideAM;
                int baseC = offsetC + i * n;

                for (int l = 0; l < k; l++)
                {
                    float valA = A[baseA + l * strideAK];
                    if (valA == 0f) continue;

                    int ptrB = offsetB + l * strideBK;
                    int ptrC = baseC;

                    for (int j = 0; j < n; j++)
                    {
                        C[ptrC] += valA * B[ptrB];
                        ptrB += strideBN;
                        ptrC++;
                    }
                }
            }
        }
    }
}