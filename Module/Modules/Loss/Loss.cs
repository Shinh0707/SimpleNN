using SimpleNN.Graph;

namespace SimpleNN.Module
{
    /// <summary>
    /// 損失関数の基底クラス.
    /// </summary>
    public abstract class Loss : Module
    {
        /// <summary>
        /// 損失値を計算する.
        /// </summary>
        /// <param name="input">モデルの出力 (Prediction)</param>
        /// <param name="target">正解ラベル (Target)</param>
        /// <returns>スカラー化された損失値</returns>
        public abstract TensorBox Forward(TensorBox input, TensorBox target);
    }
}