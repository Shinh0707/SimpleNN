namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    public class SinFunction : SingleFunction<SinFunction>
    {
         private const float eps = 0.01f;
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sin(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var cos = Tensor.Shrink(Tensor.Cos(x), eps);
            return new[] { grad * cos };
        }
    }
    public class CosFunction : SingleFunction<CosFunction>
    {
        private const float eps = 0.01f;
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Cos(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var sin = Tensor.Shrink(Tensor.Sin(x), eps);
            return new[] { grad * -sin };
        }
    }
    public class TanFunction : SingleFunction<TanFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Tan(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var sec = Tensor.Sec(x);
            return new[] { grad * Tensor.Square(sec) };
        }
    }
    public class CotFunction : SingleFunction<CotFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Cot(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var csc = Tensor.Csc(x);
            return new[] { grad * (-Tensor.Square(csc)) };
        }
    }
    public class SecFunction : SingleFunction<SecFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Sec(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var sec = ctx.Tensor;
            var tan = Tensor.Tan(x);
            return new[] { grad * sec * tan };
        }
    }
    public class CscFunction : SingleFunction<CscFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Csc(ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            var csc = ctx.Tensor;
            var cot = Tensor.Cot(x);
            return new[] { grad * (-csc * cot) };
        }
    }
}