namespace SimpleNN
{
    using System; // IDisposable のために必要
    /// <summary>
    /// プロジェクト全体のグローバル設定を管理する
    /// </summary>
    public static class Config
    {
        // 勾配計算（Grad）を有効にするかどうかの内部フラグ
        private static bool _useGrad = true;

        // NoGradScope のネストレベルを追跡する参照カウント
        // 0 の場合のみ _useGrad が true になる
        private static int _noGradScopeCount = 0;

        /// <summary>
        /// 現在のスコープで勾配計算が有効か（NoGradスコープ外か）を取得する
        /// </summary>
        public static bool UseGrad => _useGrad;

        /// <summary>
        /// IDisposable を実装したスコープ構造体.
        /// using ブロック内で勾配計算を一時的に無効化する.
        /// Config.EnterNoGradScope() 経由でのみ生成する.
        /// </summary>
        public readonly struct NoGradScope : IDisposable
        {
            // この構造体は状態を持たず, Dispose のみが呼ばれる

            /// <summary>
            /// スコープを抜け, 勾配計算の状態を復元する.
            /// ネストの最外層を抜ける場合にのみ勾配計算を有効（true）に戻す.
            /// </summary>
            public void Dispose()
            {
                _noGradScopeCount--;
                if (_noGradScopeCount == 0)
                {
                    _useGrad = true;
                }
            }
        }

        /// <summary>
        /// 勾配計算を一時的に無効にするスコープを開始する.
        /// </summary>
        /// <remarks>
        /// <code>
        /// // 元の状態
        /// Debug.Log(Config.UseGrad); // true
        /// using (Config.NoGrad())
        /// {
        ///     Debug.Log(Config.UseGrad); // false
        ///     using (Config.NoGrad())
        ///     {
        ///         Debug.Log(Config.UseGrad); // false
        ///     }
        ///     Debug.Log(Config.UseGrad); // false
        /// }
        /// Debug.Log(Config.UseGrad); // true
        /// </code>
        /// </remarks>
        /// <returns>
        /// using ブロックで破棄可能な NoGradScope インスタンス.
        /// </returns>
        public static NoGradScope NoGrad()
        {
            _noGradScopeCount++;
            _useGrad = false; // ネストレベルに関わらず false に設定
            return new NoGradScope();
        }
    }
}