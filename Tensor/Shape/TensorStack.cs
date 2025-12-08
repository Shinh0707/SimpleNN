namespace SimpleNN.Tensor
{
    using System;
    using SimpleNN.Util;

    public partial class Tensor
    {
        /// <summary>
        /// Concatenates a sequence of tensors along a new dimension.
        /// All tensors need to be of the same size.
        /// </summary>
        public static Tensor Stack(int dim, params Tensor[] tensors)
        {
            if (tensors == null || tensors.Length == 0)
            {
                throw new ArgumentException("Tensors array cannot be empty or null.");
            }

            var first = tensors[0];
            int ndim = first.NDim;
            // Result will have ndim + 1 dimensions
            int outNDim = ndim + 1;

            if (dim < 0) dim += outNDim;
            if (dim < 0 || dim >= outNDim)
            {
                throw new ArgumentOutOfRangeException(nameof(dim), $"Dimension {dim} is out of range for output tensor with {outNDim} dimensions.");
            }

            // Validate shapes
            for (int i = 1; i < tensors.Length; i++)
            {
                if (!IsSameSize(first, tensors[i]))
                {
                    throw new ArgumentException($"All tensors must have the same size. Tensor at index {i} has different size.");
                }
            }

            // Calculate output size
            int[] outSize = new int[outNDim];
            // Copy sizes before dim
            for (int i = 0; i < dim; i++) outSize[i] = first.Size[i];
            // Insert stacked dimension size
            outSize[dim] = tensors.Length;
            // Copy sizes after dim
            for (int i = dim; i < ndim; i++) outSize[i + 1] = first.Size[i];

            var outTensor = new Tensor(new float[Util.ArrayExt.IntProd(outSize)], outSize);

            // Optimization: If dim is 0, we can just copy contiguous blocks sequentially
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
            // Similar to Concat, but simpler because all inputs are same size.
            // We can view the copy as: Outer Loop -> Stack Axis -> Inner Loop (Block Copy)
            
            // Outer size: product of dims before 'dim' in INPUT tensor
            int outerSize = 1;
            for (int d = 0; d < dim; d++) outerSize *= first.Size[d];

            // Inner size: product of dims after 'dim' in INPUT tensor
            int innerSize = 1;
            for (int d = dim; d < ndim; d++) innerSize *= first.Size[d];

            // Pre-fetch contiguous data
            float[][] srcDatas = new float[tensors.Length][];
            for(int i=0; i<tensors.Length; i++)
            {
                srcDatas[i] = tensors[i].IsContiguous ? tensors[i].Data : tensors[i].GetContiguousData();
            }

            int outOffset = 0;
            int blockSize = innerSize; // Each tensor contributes 'innerSize' elements per outer iteration
            int[] srcOffsets = new int[tensors.Length]; // Starts at 0

            for (int o = 0; o < outerSize; o++)
            {
                for (int i = 0; i < tensors.Length; i++)
                {
                    Array.Copy(srcDatas[i], srcOffsets[i], outTensor.Data, outOffset, blockSize);
                    srcOffsets[i] += blockSize;
                    outOffset += blockSize;
                }
            }

            return outTensor;
        }

        /// <summary>
        /// Splits a tensor into a list of tensors along a dimension, removing that dimension.
        /// Inverse of Stack.
        /// </summary>
        public static Tensor[] Unstack(int dim, Tensor tensor)
        {
            if (tensor is null) throw new ArgumentNullException(nameof(tensor));

            int ndim = tensor.NDim;
            if (dim < 0) dim += ndim;
            if (dim < 0 || dim >= ndim) throw new ArgumentOutOfRangeException(nameof(dim));

            int numSplits = tensor.Size[dim];
            Tensor[] result = new Tensor[numSplits];

            // Output size is input size with 'dim' removed
            int[] outSize = new int[ndim - 1];
            for (int i = 0; i < dim; i++) outSize[i] = tensor.Size[i];
            for (int i = dim + 1; i < ndim; i++) outSize[i - 1] = tensor.Size[i];

            // Initialize result tensors
            for (int i = 0; i < numSplits; i++)
            {
                result[i] = new Tensor(new float[Util.ArrayExt.IntProd(outSize)], outSize);
            }

            // Prepare source data
            float[] srcData = tensor.IsContiguous ? tensor.Data : tensor.GetContiguousData();

            // Optimization: If dim is 0, we are just splitting the array into chunks
            if (dim == 0)
            {
                int blockSize = srcData.Length / numSplits;
                int offset = 0;
                for (int i = 0; i < numSplits; i++)
                {
                    Array.Copy(srcData, offset, result[i].Data, 0, blockSize);
                    offset += blockSize;
                }
                return result;
            }

            // General case
            // Outer size: product of dims before 'dim'
            int outerSize = 1;
            for (int d = 0; d < dim; d++) outerSize *= tensor.Size[d];

            // Inner size: product of dims after 'dim'
            int innerSize = 1;
            for (int d = dim + 1; d < ndim; d++) innerSize *= tensor.Size[d];

            int srcOffset = 0;
            int[] dstOffsets = new int[numSplits]; // Starts at 0
            int copySize = innerSize;

            for (int o = 0; o < outerSize; o++)
            {
                for (int i = 0; i < numSplits; i++)
                {
                    Array.Copy(srcData, srcOffset, result[i].Data, dstOffsets[i], copySize);
                    srcOffset += copySize;
                    dstOffsets[i] += copySize;
                }
            }

            return result;
        }
    }
}
