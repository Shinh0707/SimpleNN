namespace SimpleNN.Graph.Functions
{
    using UnityEngine;
    using SimpleNN.Tensor;

    /// <summary>
    /// 自然対数 (Natural Logarithm) 関数.
    /// </summary>
    public class LogFunction : SingleFunction<LogFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Log(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            //Debug.Log($"[Backward] log x: x={x}, res={grad}/{x}");
            return new[] { grad / x};
        }
    }
}