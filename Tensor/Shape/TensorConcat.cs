namespace SimpleNN.Tensor
{
    using System;
    using SimpleNN.Util;

    public partial class Tensor
    {
        public static Tensor Concat(int dim, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException("Tensors array cannot be empty or null.");
            }

            var first = tensors[0];
            int ndim = first.NDim;
            if (dim < 0) dim += ndim;

            if (dim < 0 || dim >= ndim)
            {
                throw new ArgumentOutOfRangeException(nameof(dim), $"Dimension {dim} is out of range for tensor with {ndim} dimensions.");
            }

            // Validate shapes and calculate output size
            int[] outSize = (int[])first.Size.Clone();
            int sumDim = 0;

            for (int i = 0; i < tensors.Length; i++)
            {
                var t = tensors[i];
                if (t.NDim != ndim)
                {
                    throw new ArgumentException($"All tensors must have the same number of dimensions. Expected {ndim}, got {t.NDim} at index {i}.");
                }

                for (int d = 0; d < ndim; d++)
                {
                    if (d != dim && t.Size[d] != outSize[d])
                    {
                        throw new ArgumentException($"Sizes must match at dimension {d} (expected {outSize[d]}, got {t.Size[d]} at index {i}).");
                    }
                }
                sumDim += t.Size[dim];
            }

            outSize[dim] = sumDim;
            var outTensor = new Tensor(new float[Util.ArrayExt.IntProd(outSize)], outSize);
            
            // Optimization: If dim is 0, we can just copy contiguous blocks if inputs are contiguous
            if (dim == 0)
            {
                int offset = 0;
                for (int i = 0; i < tensors.Length; i++)
                {
                    var t = tensors[i];
                    var tData = t.IsContiguous ? t.Data : t.GetContiguousData();
                    Array.Copy(tData, 0, outTensor.Data, offset, tData.Length);
                    offset += tData.Length;
                }
                return outTensor;
            }

            // General case
            // We can view the copy as: Outer Loop -> Concat Axis -> Inner Loop (Block Copy)
            // Outer size: product of dims before 'dim'
            // Inner size: product of dims after 'dim'
            
            int outerSize = 1;
            for (int d = 0; d < dim; d++) outerSize *= outSize[d];

            int innerSize = 1;
            for (int d = dim + 1; d < ndim; d++) innerSize *= outSize[d];

            // For the output tensor, the stride for the concat dimension is innerSize.
            // But wait, we are filling a contiguous array.
            // The structure in memory for contiguous tensor is:
            // [Outer][Dim][Inner]
            
            // We iterate Outer.
            //   For each tensor t:
            //     Copy t.Size[dim] * innerSize elements.
            
            int outOffset = 0;
            
            // Pre-calculate contiguous data for all tensors to avoid repeated checks/allocations inside loop?
            // No, GetContiguousData returns full array.
            // If we use GetContiguousData, we can treat them as flat arrays.
            
            float[][] srcDatas = new float[tensors.Length][];
            for(int i=0; i<tensors.Length; i++)
            {
                srcDatas[i] = tensors[i].IsContiguous ? tensors[i].Data : tensors[i].GetContiguousData();
            }

            // To optimize, we can loop:
            // for outer in 0..outerSize
            //   for t in tensors
            //     copy t.Size[dim] * innerSize elements
            
            // The size of block to copy for tensor t is t.Size[dim] * innerSize
            // The source offset for tensor t increases by block size each outer iteration.
            
            int[] blockSizes = new int[tensors.Length];
            for(int i=0; i<tensors.Length; i++)
            {
                blockSizes[i] = tensors[i].Size[dim] * innerSize;
            }

            int[] srcOffsets = new int[tensors.Length]; // Starts at 0

            for (int o = 0; o < outerSize; o++)
            {
                for (int i = 0; i < tensors.Length; i++)
                {
                    int count = blockSizes[i];
                    Array.Copy(srcDatas[i], srcOffsets[i], outTensor.Data, outOffset, count);
                    srcOffsets[i] += count;
                    outOffset += count;
                }
            }

            return outTensor;
        }
    }
}
