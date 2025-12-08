namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    public class FlattenFunction : KwargsFunction<FlattenFunction, FlattenFunction.Kwargs>
    {
        public class Kwargs
        {
            public int startIndex = 0;
            public int endIndex = -1;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Flatten(ctx.GetInput(0), kwargs.startIndex, kwargs.endIndex);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            var inputSize = ctx.GetInput(0).Size;
            return new[] { Tensor.Reshape(grad, inputSize) };
        }
    }
}