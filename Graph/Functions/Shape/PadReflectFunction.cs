namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    public class PadReflectFunction : KwargsFunction<PadReflectFunction, PadReflectFunction.Kwargs>
    {
        public class Kwargs
        {
            public (int left, int right)[] paddingSizes;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.ReflectPad(ctx.GetInput(0), kwargs.paddingSizes);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { Tensor.PadBackward(grad, kwargs.paddingSizes, PaddingMode.REFLECT) };
        }
    }
}
