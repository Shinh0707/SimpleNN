namespace SimpleNN.Tensor
{
    using System;

    public partial class Tensor
    {
        public static Tensor[] Split(int dim, Tensor tensor, int splitSize)
        {
            if (tensor is null) throw new ArgumentNullException(nameof(tensor));
            
            int ndim = tensor.NDim;
            if (dim < 0) dim += ndim;
            if (dim < 0 || dim >= ndim) throw new ArgumentOutOfRangeException(nameof(dim));

            int dimSize = tensor.Size[dim];
            if (splitSize <= 0) throw new ArgumentException("Split size must be positive.");
            
            int numSplits = (dimSize + splitSize - 1) / splitSize;
            int[] splitSizes = new int[numSplits];
            for(int i=0; i<numSplits; i++)
            {
                splitSizes[i] = Math.Min(splitSize, dimSize - i * splitSize);
            }

            return Split(dim, tensor, splitSizes);
        }

        public static Tensor[] Split(int dim, Tensor tensor, params int[] splitSizes)
        {
            if (tensor is null) throw new ArgumentNullException(nameof(tensor));
            
            int ndim = tensor.NDim;
            if (dim < 0) dim += ndim;
            if (dim < 0 || dim >= ndim) throw new ArgumentOutOfRangeException(nameof(dim));

            int dimSize = tensor.Size[dim];
            long sum = 0;
            for(int i=0; i<splitSizes.Length; i++) sum += splitSizes[i];
            if (sum != dimSize) throw new ArgumentException($"Sum of split sizes ({sum}) must match tensor dimension size ({dimSize}).");

            Tensor[] result = new Tensor[splitSizes.Length];
            
            // Prepare source data
            float[] srcData = tensor.IsContiguous ? tensor.Data : tensor.GetContiguousData();

            int outerSize = 1;
            for (int d = 0; d < dim; d++) outerSize *= tensor.Size[d];

            int innerSize = 1;
            for (int d = dim + 1; d < ndim; d++) innerSize *= tensor.Size[d];

            // Initialize result tensors
            for (int i = 0; i < splitSizes.Length; i++)
            {
                int[] newSize = (int[])tensor.Size.Clone();
                newSize[dim] = splitSizes[i];
                result[i] = new Tensor(new float[Util.ArrayExt.IntProd(newSize)], newSize);
            }

            int srcOffset = 0;
            int[] dstOffsets = new int[splitSizes.Length]; // Starts at 0

            // Loop structure similar to Concat but reversed
            // Outer -> Splits -> Inner
            
            for (int o = 0; o < outerSize; o++)
            {
                for (int i = 0; i < splitSizes.Length; i++)
                {
                    int blockSize = splitSizes[i] * innerSize;
                    Array.Copy(srcData, srcOffset, result[i].Data, dstOffsets[i], blockSize);
                    srcOffset += blockSize;
                    dstOffsets[i] += blockSize;
                }
            }

            return result;
        }
    }
}
