namespace SimpleNN.Graph.Functions
{
    using System.Collections.Generic;
    using SimpleNN.Tensor;

    public class StackArgs
    {
        public int dim;
    }

    public class StackFunction : KwargsFunction<StackFunction, StackArgs>
    {
        internal override Tensor Forward(StackArgs args, Context ctx)
        {
            var tensors = new List<Tensor>();
            int i = 0;
            while (ctx.TryGetInput(i, out Tensor t))
            {
                tensors.Add(t);
                i++;
            }
            return Tensor.Stack(args.dim, tensors.ToArray());
        }

        internal override Tensor[] Backward(StackArgs args, Context ctx, Tensor grad)
        {
            var grads = Tensor.Unstack(args.dim, grad);
            return grads;
        }
    }
}
