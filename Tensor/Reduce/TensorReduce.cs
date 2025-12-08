using System;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        private static void PrepareReduceFunction(
            Tensor tensor, int dim, bool keepDims,
            out int ndim, out float[] outData, out int[] outSize
        )
        {
            ndim = tensor.NDim;
            if (dim < 0) dim += ndim;
            if (dim < 0 || dim >= ndim) throw new ArgumentOutOfRangeException(nameof(dim));
            if (keepDims)
            {
                outSize = new int[ndim];
                Array.Copy(tensor.Size, outSize, ndim);
                outSize[dim] = 1;
            }
            else
            {
                outSize = new int[ndim - 1];
                int idx = 0;
                for (int i = 0; i < ndim; i++)
                {
                    if (i != dim) outSize[idx++] = tensor.Size[i];
                }
            }
            int outLen = 1;
            for (int i = 0; i < outSize.Length; i++) outLen *= outSize[i];
            outData = new float[outLen];
        }
        private delegate float ReduceFunc(float inData, float reducedValue);
        private static Tensor ReduceFunction(
            Tensor tensor, int dim, bool keepDims,
            ReduceFunc func, float? startValue = null
        )
        {
            PrepareReduceFunction(tensor, dim, keepDims, out int ndim, out var outData, out var outSize);
            if (startValue.HasValue)
            {
                Array.Fill(outData, startValue.Value);
            }
            var inStrides = tensor.Strides; // プロパティ経由で取得(正当性保証)
            var inSize = tensor.Size;
            var inData = tensor._data;
            var reducedStrides = new int[ndim];

            int currentOutStride = 1;
            for (int i = ndim - 1; i >= 0; i--)
            {
                if (i == dim)
                {
                    reducedStrides[i] = 0;
                }
                else
                {
                    reducedStrides[i] = currentOutStride;
                    currentOutStride *= inSize[i];
                }
            }
            var totalElements = tensor.TotalSize;
            int[] counters = new int[ndim];

            for (uint i = 0; i < totalElements; i++)
            {
                int inPtr = 0;
                int outPtr = 0;

                for (int d = 0; d < ndim; d++)
                {
                    inPtr += counters[d] * inStrides[d];
                    outPtr += counters[d] * reducedStrides[d];
                }
                outData[outPtr] = func(inData[inPtr], outData[outPtr]);
                for (int d = ndim - 1; d >= 0; d--)
                {
                    counters[d]++;
                    if (counters[d] < inSize[d]) break;
                    counters[d] = 0;
                }
            }
            return new Tensor(outData, outSize);
        }
        private static Tensor ReduceFunction(
            Tensor tensor, ReduceFunc func, float? startValue = null
        )
        {
            var t = tensor;
            var d = t.NDim;
            for(int i = 0; i < d; i++)
            {
                t = ReduceFunction(t,0,false,func,startValue);
            }
            return t;
        }
    }
}