namespace SimpleNN.Optimizer
{
    using System.Collections.Generic;
    using SimpleNN.Graph;
    using SimpleNN.Tensor;

    /// <summary>
    /// Adadelta オプティマイザ.
    /// Adagradの学習率が単調減少する問題を解決するために提案された手法.
    /// グローバルな学習率を持たず, 更新量の単位がパラメータと一致するという特徴を持つ.
    /// </summary>
    public class Adadelta : Optimizer
    {
        private readonly float _rho;
        private readonly float _epsilon;

        // (squareAvg: 勾配の二乗移動平均, accDelta: 更新量の二乗移動平均)
        // 状態をタプルの配列で管理する
        private readonly (Tensor squareAvg, Tensor accDelta)[] _states;

        /// <summary>
        /// Adadelta を初期化する.
        /// </summary>
        /// <param name="rho">減衰率 (例: 0.90 〜 0.95)</param>
        /// <param name="epsilon">ゼロ除算を防ぐための微小値</param>
        /// <param name="parameters">最適化対象のパラメータ</param>
        public Adadelta(float rho, float epsilon, params List<TensorBox>[] parameters) 
            : base(parameters)
        {
            _rho = rho;
            _epsilon = epsilon;
            _states = new (Tensor squareAvg, Tensor accDelta)[NumParameters];
        }

        /// <summary>
        /// デフォルト設定 (rho=0.9, epsilon=1e-6) で Adadelta を初期化する.
        /// </summary>
        public Adadelta(params List<TensorBox>[] parameters)
            : this(0.9f, 1e-6f, parameters)
        {
        }

        /// <summary>
        /// パラメータ更新計算を行う.
        /// Adadeltaの更新式:
        /// 1. E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
        /// 2. RMS[g]_t = sqrt(E[g^2]_t + eps)
        /// 3. delta_x = - (RMS[dx]_{t-1} / RMS[g]_t) * g
        /// 4. E[dx^2]_t = rho * E[dx^2]_{t-1} + (1 - rho) * delta_x^2
        /// 5. x_{t+1} = x_t + delta_x
        /// </summary>
        protected override Tensor StepGrad(int id, Tensor tensor, Tensor grad)
        {
            var state = _states[id];

            // 初回初期化
            if (state.squareAvg is null)
            {
                state.squareAvg = grad * 0.0f; // ゼロテンソルを作成
                state.accDelta = grad * 0.0f;
            }

            var sqAvg = state.squareAvg;
            var accDelta = state.accDelta;

            // 1. 勾配の二乗移動平均を更新 (E[g^2])
            sqAvg = (sqAvg * _rho) + (Tensor.Square(grad) * (1.0f - _rho));

            // 2. 勾配のRMSを計算
            var std = Tensor.Sqrt(sqAvg + _epsilon);

            // 3. 前回の更新量のRMSを計算
            var deltaStd = Tensor.Sqrt(accDelta + _epsilon);

            // 更新量を計算 (update = -delta_x の大きさ)
            // 公式: delta_x = - (RMS[dx] / RMS[g]) * g
            var update = deltaStd * grad / std;

            // 4. 更新量の二乗移動平均を更新 (E[dx^2])
            // 次のステップで利用するために, 今回の更新量を蓄積する
            accDelta = (accDelta * _rho) + (Tensor.Square(update) * (1.0f - _rho));

            // 状態を保存
            _states[id] = (sqAvg, accDelta);

            // 5. パラメータ更新 (tensor - update)
            return tensor - update;
        }
    }
}