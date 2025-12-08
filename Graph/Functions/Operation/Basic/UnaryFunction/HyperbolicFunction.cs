namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Hyperbolic Tangent 関数.
    /// </summary>
    public class TanhFunction : SingleFunction<TanhFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Tanh(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            // y = tanh(x)
            // dy/dx = 1 - y^2
            
            var y = ctx.Tensor;
            
            // y^2
            var y2 = Tensor.Square(y);
            
            // 1 - y^2
            var d = Tensor.Sub_(1.0f, y2);
            
            return new[] { Tensor.Mul_(grad, d) };
        }
    }
    public class SechFunction : SingleFunction<SechFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sech(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            // y = sech(x)
            // dy/dx = -tanh(x) * sech(x) = -tanh(x) * y
            
            var x = ctx.GetInput(0);
            var y = ctx.Tensor;
            
            // tanh(x)
            var tanh = Tensor.Tanh(x);
            
            // -tanh(x)
            var negTanh = Tensor.Neg(tanh);
            
            // -tanh(x) * y
            var d = Tensor.Mul_(negTanh, y);
            
            return new[] { Tensor.Mul_(grad, d) };
        }
    }
}