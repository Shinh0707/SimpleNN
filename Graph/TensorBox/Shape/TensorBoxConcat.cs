namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;

    public partial class TensorBox
    {
        public static TensorBox Concat(int dim, params TensorBox[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new System.ArgumentException("Boxes array cannot be empty or null.");
            }

            var ctxs = new Context[tensors.Length];
            for (int i = 0; i < tensors.Length; i++)
            {
                ctxs[i] = tensors[i]._ctx;
            }

            var args = new ConcatArgs { dim = dim };
            var newCtx = ConcatFunction.Forward(args, ctxs);
            return new TensorBox(newCtx);
        }
        
        public static TensorBox Stack(int dim, params TensorBox[] tensors)
        {
            var ctxs = new Context[tensors.Length];
            for (int i = 0; i < tensors.Length; i++)
            {
                ctxs[i] = tensors[i]._ctx;
            }
            var args = new StackArgs { dim = dim };
            var newCtx = StackFunction.Forward(args, ctxs);
            return new TensorBox(newCtx);
        }
    }
}
