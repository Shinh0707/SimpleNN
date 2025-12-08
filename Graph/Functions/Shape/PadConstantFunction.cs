namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;

    public class PadConstantFunction : KwargsFunction<PadConstantFunction, PadConstantFunction.Kwargs>
    {
        public class Kwargs
        {
            public (int left, int right)[] paddingSizes;
            public float value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.ConstantPad(ctx.GetInput(0), kwargs.paddingSizes, kwargs.value);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { Tensor.PadBackward(grad, kwargs.paddingSizes, PaddingMode.CONSTANT) };
        }
    }
}
