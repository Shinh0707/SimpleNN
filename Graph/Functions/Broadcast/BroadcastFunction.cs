namespace SimpleNN.Graph.Functions
{
    using SimpleNN.Tensor;
    using UnityEngine;

    public class BroadcastFunction : KwargsFunction<BroadcastFunction, BroadcastFunction.Kwargs>
    {
        public class Kwargs
        {
            public int dim;
            public int targetSize;
        }
        internal override Tensor Forward(Kwargs kwargs, Context ctx)
        {
            //Debug.Log($"(Broadcast) Forward : TargetSize: {kwargs.targetSize} at dim{kwargs.dim}, Input={Util.StringExt.ArrayString("InputSize",ctx.GetInput(0).Size)}");
            return Tensor.Broadcast(ctx.GetInput(0), kwargs.dim, kwargs.targetSize);
        }
        internal override Tensor[] Backward(Kwargs kwargs, Context ctx, Tensor grad)
        {
            //Debug.Log($"(Broadcast) Backward : TargetSize: {kwargs.targetSize} at dim{kwargs.dim}, ForwardInput={Util.StringExt.ArrayString("InputSize",ctx.GetInput(0).Size)}, GradInput={grad}");
            return new[] { Tensor.Sum(grad, kwargs.dim, kwargs.targetSize != 0) };
        }
    }
}