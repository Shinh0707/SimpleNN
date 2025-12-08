namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public class AddFunction : SingleFunction<AddFunction>
    {
        internal override Tensor Forward(Context ctx)
        {
            return Tensor.Add_(ctx.GetInput(0), ctx.GetInput(1));
        }
        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            return new[] { grad, grad };
        }
    }
    public class AddSclFunction : KwargsFunction<AddSclFunction, AddSclFunction.Kwargs>
    {
        public class Kwargs
        {
            public float Value;
        }

        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            return Tensor.Add_(ctx.GetInput(0), kwargs.Value);
        }

        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            return new[] { grad };
        }
    }
}