namespace SimpleNN.Tensor.Audio
{
    using System;

    // FFTの計算済み定数を保持するクラス
    public class FFTPlan
    {
        public readonly int N;
        public readonly int[] BitReverseTable;
        public readonly float[] CosTable;
        public readonly float[] SinTable;

        public FFTPlan(int n)
        {
            if ((n & (n - 1)) != 0) throw new ArgumentException("N must be power of 2");
            N = n;

            // 1. ビット反転テーブルの事前計算
            BitReverseTable = new int[n];
            int m = (int)Math.Log(n, 2);
            for (int i = 0; i < n; i++)
            {
                int j = 0;
                int k = i;
                for (int l = 0; l < m; l++)
                {
                    j = (j << 1) | (k & 1);
                    k >>= 1;
                }
                BitReverseTable[i] = j;
            }

            // 2. 回転因子(Twiddle Factors)の事前計算
            // n/2個の複素数があれば十分
            int halfN = n / 2;
            CosTable = new float[halfN];
            SinTable = new float[halfN];

            // -2π/N * k
            double factor = -2.0 * Math.PI / n;
            for (int k = 0; k < halfN; k++)
            {
                // 倍精度で計算して精度確保
                CosTable[k] = (float)Math.Cos(factor * k);
                SinTable[k] = (float)Math.Sin(factor * k);
            }
        }
    }
}