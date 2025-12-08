namespace SimpleNN.Graph.Functions
{
    using System.Collections.Generic;
    using SimpleNN.Tensor;

    public class ConcatArgs
    {
        public int dim;
    }

    public class ConcatFunction : KwargsFunction<ConcatFunction, ConcatArgs>
    {
        internal override Tensor Forward(ConcatArgs args, Context ctx)
        {
            var tensors = new List<Tensor>();
            var splitSizes = new List<int>();
            int i = 0;
            while (ctx.TryGetInput(i, out Tensor t))
            {
                tensors.Add(t);
                // Handle negative dim for size extraction
                int d = args.dim;
                if (d < 0) d += t.NDim;
                splitSizes.Add(t.Size[d]);
                i++;
            }

            ctx.RegisterDatas(splitSizes.ToArray());
            
            return Tensor.Concat(args.dim, tensors.ToArray());
        }

        internal override Tensor[] Backward(ConcatArgs args, Context ctx, Tensor grad)
        {
            var splitSizes = ctx.GetRegisteredData<int[]>(0);
            return Tensor.Split(args.dim, grad, splitSizes);
        }
    }
}
