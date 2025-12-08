namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    /// <summary>
    /// Division Function (a / b).
    /// </summary>
    public class DivFunction : SingleFunction<DivFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Div_(ctx.GetInput(0), ctx.GetInput(1));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);

            // da = grad / b
            var da = Tensor.Div_(grad, b);

            // db = -grad * a / (b * b)

            // b^2
            var b2 = Tensor.Square(b);

            // a / b^2
            var term = Tensor.Div_(a, b2);

            // -grad
            var negGrad = Tensor.Neg(grad);

            // -grad * (a / b^2)
            var db = Tensor.Mul_(negGrad, term);

            return new[] { da, db };
        }
    }
    public class DivRSclFunction : KwargsFunction<DivRSclFunction, DivRSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Div_(ctx.GetInput(0), kwargs.Value);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { grad / kwargs.Value };
        }
    }
    public class DivLSclFunction : KwargsFunction<DivLSclFunction, DivLSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Div_(kwargs.Value, ctx.GetInput(0));
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var x = ctx.GetInput(0);
            return new[] { ((-grad) * kwargs.Value) / Tensor.Square(x)};
        }
    }
}