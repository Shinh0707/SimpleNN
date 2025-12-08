namespace SimpleNN.Util
{
    using UnityEngine;
    public static partial class ArrayExt
    {
        public static void SetSeed(int seed) => UnityEngine.Random.InitState(seed);
        /// <summary>
        /// 指定された範囲内で <paramref name="n"/> 個のランダムな浮動小数点数を生成する.
        /// </summary>
        /// <param name="n">生成する数値の個数</param>
        /// <param name="min">乱数の最小値（含む）</param>
        /// <param name="max">乱数の最大値（含む）</param>
        /// <returns>生成された <paramref name="n"/> 個のランダムな浮動小数点数を含む配列</returns>
        public static float[] Random(int n, float min, float max)
        {
            float[] results = new float[n];
            
            for (int i = 0; i < n; ++i)
            {
                results[i] = UnityEngine.Random.Range(min, max);
            }
            
            return results;
        }

        /// <summary>
        /// 平均 <paramref name="loc"/>, 標準偏差 <paramref name="std"/> の
        /// 正規分布（ガウス分布）に従う <paramref name="n"/> 個のランダムな浮動小数点数を生成する.
        /// </summary>
        /// <remarks>
        /// ボックス＝ミュラー（Box-Muller）法を使用して計算する.
        /// </remarks>
        /// <param name="n">生成する数値の個数</param>
        /// <param name="loc">分布の平均（位置）</param>
        /// <param name="std">分布の標準偏差</param>
        /// <returns>
        /// 生成された <paramref name="n"/> 個の正規分布乱数を含む配列
        /// </returns>
        public static float[] Normal(int n, float loc, float std)
        {
            float[] results = new float[n];

            for (int i = 0; i < n; i += 2)
            {
                float u1;
                do
                {
                    u1 = UnityEngine.Random.value;
                } while (u1 == 0.0f);
                
                float u2 = UnityEngine.Random.value;

                float r = Mathf.Sqrt(-2.0f * Mathf.Log(u1));
                float theta = 2.0f * Mathf.PI * u2;
                
                float z1 = r * Mathf.Cos(theta);
                float z2 = r * Mathf.Sin(theta);

                results[i] = loc + std * z1;

                if (i + 1 < n)
                {
                    results[i + 1] = loc + std * z2;
                }
            }
            
            return results;
        }
    }
}