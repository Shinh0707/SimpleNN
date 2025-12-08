namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    public abstract class ReduceKwargs
    {
        public int dim;
        public int Dim(int ndim) => (dim < 0) ? ndim + dim : dim;
        public bool keepDims;
    }
    public abstract class ReduceFunction<T,U> : Function<T> where T : ReduceFunction<T,U>, new() where U : ReduceKwargs, new()
    {
        private U _kwargs;
        public static Context Forward(U kwargs, params Tensor[] inputTensors)
        {
            var inputCtx = new Context[inputTensors.Length];
            for (int i = 0; i < inputTensors.Length; i++)
            {
                inputCtx[i] = new(inputTensors[i]);
            }
            return Forward(kwargs, inputCtx);
        }
        public static Context Forward(U kwargs, params Context[] inputCtx)
        {
            var func = new T
            {
                _kwargs = kwargs ?? new()
            };
            return new Context(inputCtx, func.Forward, func.Backward);
        }
        internal override Tensor Forward(Context ctx)
        {
            return Forward(_kwargs, ctx);
        }
        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var inputSize = ctx.GetInput(0).Size;
            var gradTensor = grad;
            if (!_kwargs.keepDims)
            {
                gradTensor = Tensor.Unsqueeze(gradTensor, _kwargs.dim);
            }
            gradTensor = Tensor.Reshape(Tensor.Broadcast(gradTensor, inputSize), inputSize);
            return Backward(_kwargs, ctx, gradTensor);
        }
        internal abstract Tensor Forward(U kwargs,Context ctx);
        internal abstract Tensor[] Backward(U kwargs, Context ctx, Tensor grad);
    }
    public abstract class ReduceFunction<T> : SingleFunction<T> where T : ReduceFunction<T>, new()
    {
        internal override Tensor[] Backward(Context ctx, Tensor grad)
        {
            var inputSize = ctx.GetInput(0).Size;
            var gradTensor = grad;
            gradTensor = Tensor.Reshape(Tensor.Broadcast(gradTensor, inputSize), inputSize);
            return ReduceBackward(ctx, gradTensor);
        }
        internal abstract Tensor[] ReduceBackward(Context ctx, Tensor grad);
    }
}