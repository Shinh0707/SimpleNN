namespace SimpleNN.Graph
{
    using System;
    using SimpleNN.Graph.Functions;
    using SimpleNN.Tensor;
    public partial class TensorBox
    {
        private static bool TryBroadcastCtx(Context ctx,int dim, int targetSize, out Context newCtx)
        {
            int sz = ctx.Tensor.Size[dim];
            newCtx = ctx;
            if (targetSize == sz) return false;
            
            // サイズが1以外で、かつターゲットと異なる場合はブロードキャスト不可
            if ((sz != 1) && (targetSize != 1)) 
                throw new ArgumentException($"Cannot broadcast dimension {dim} with size {sz} to {targetSize}.");
            
            newCtx = BroadcastFunction.Forward(new()
            {
                dim = dim,
                targetSize = targetSize
            }, new[] { ctx });
            return true;
        }
        private static (Context broadA, Context broadB) BroadcastBothCtx(TensorBox a, TensorBox b)
        {
            Context ctxA = a._ctx;
            Context ctxB = b._ctx;
            int aDim = a.NDim;
            int bDim = b.NDim;

            if (aDim != bDim)
            {
                if (aDim > bDim)
                {
                    int diff = aDim - bDim;
                    int[] newShape = new int[aDim];
                    for (int i = 0; i < diff; i++) newShape[i] = 1;
                    Array.Copy(b.Size, 0, newShape, diff, bDim);

                    ctxB = Reshape(ctxB, newShape);
                }
                else
                {
                    int diff = bDim - aDim;
                    int[] newShape = new int[bDim];
                    for (int i = 0; i < diff; i++) newShape[i] = 1;
                    Array.Copy(a.Size, 0, newShape, diff, aDim);

                    ctxA = Reshape(ctxA, newShape);
                    aDim = bDim;
                }
            }

            int ndim = aDim;
            var aSize = ctxA.Tensor.Size;
            var bSize = ctxB.Tensor.Size;

            for (int dim = ndim - 1; dim >= 0; dim--)
            {
                int asz = aSize[dim];
                int bsz = bSize[dim];

                if (asz == bsz) continue;

                if ((asz != 1) && (bsz != 1))
                    throw new ArgumentException($"Operands could not be broadcast together with shapes {string.Join(",", aSize)} and {string.Join(",", bSize)}");

                if (asz == 1)
                {
                    ctxA = BroadcastFunction.Forward(new()
                    {
                        dim = dim,
                        targetSize = bsz
                    }, new[] { ctxA });
                }
                else
                {
                    ctxB = BroadcastFunction.Forward(new()
                    {
                        dim = dim,
                        targetSize = asz
                    }, new[] { ctxB });
                }
            }

            return (ctxA, ctxB);
        }
        public static (TensorBox broadA, TensorBox broadB) BroadcastBoth(TensorBox a, TensorBox b)
        {
            (var bA, var bB) = BroadcastBothCtx(a, b);
            return (new(bA), new(bB));
        }
        public TensorBox Broadcast(params int[] targetSize)
        {
            int ndim = targetSize.Length;
            var ctx = _ctx;
            int tsz;
            bool broadcasted = false;
            for (int dim = ndim - 1; dim >= 0; dim--)
            {
                tsz = targetSize[dim];
                broadcasted = TryBroadcastCtx(ctx, dim, tsz, out ctx);
            }
            if (broadcasted)
            {
                return new(ctx);
            }
            return this;
        }
        public TensorBox Broadcast(Tensor tensor) => Broadcast(tensor.Size);
        public TensorBox Broadcast(TensorBox tensor) => Broadcast(tensor.Size);
        private Context BroadcastCtx(int[] targetSize)
        {
            int ndim = targetSize.Length;
            var ctx = _ctx;
            int tsz;
            bool broadcasted = false;
            for (int dim = ndim - 1; dim >= 0; dim--)
            {
                tsz = targetSize[dim];
                broadcasted = TryBroadcastCtx(ctx, dim, tsz, out ctx);
            }
            if (broadcasted)
            {
                return ctx;
            }
            return _ctx;
        }
        private Context BroadcastCtx(Tensor target) => BroadcastCtx(target.Size);
        private Context BroadcastCtx(Context target) => BroadcastCtx(target.Tensor);
        private Context BroadcastCtx(TensorBox target) => BroadcastCtx(target.Size);
    }
}