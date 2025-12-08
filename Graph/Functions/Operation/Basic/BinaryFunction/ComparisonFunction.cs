namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class MaximumFunction : SingleFunction<MaximumFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Maximum_(ctx.GetInput(0), ctx.GetInput(1));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);

            var maskA = a >= b;
            var maskB = b >= a;

            return new[] { grad * maskA, grad * maskB };
        }
    }

    public class MaximumScalerFunction : KwargsFunction<MaximumScalerFunction, MaximumScalerFunction.Kwargs>
    {
        public class Kwargs { public float Scalar; }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Maximum(ctx.GetInput(0), kwargs.Scalar);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var mask = a >= kwargs.Scalar;
            return new[] { grad * mask };
        }
    }

    public class MinimumFunction : SingleFunction<MinimumFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Minimum_(ctx.GetInput(0), ctx.GetInput(1));
        }

        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var b = ctx.GetInput(1);

            var maskA = a <= b;
            var maskB = b <= a;

            return new[] { grad * maskA, grad * maskB };
        }
    }

    public class MinimumScalerFunction : KwargsFunction<MinimumScalerFunction, MinimumScalerFunction.Kwargs>
    {
        public class Kwargs { public float Scalar; }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Minimum(ctx.GetInput(0), kwargs.Scalar);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var a = ctx.GetInput(0);
            var mask = a <= kwargs.Scalar;
            return new[] { grad * mask };
        }
    }
}