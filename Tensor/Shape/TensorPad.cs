namespace SimpleNN.Tensor
{
    using UnityEngine;
    public enum PaddingMode
    {
        CONSTANT, REFLECT, REPLICATE, CIRCULAR
    }
    public partial class Tensor
    {
        public static Tensor Pad(Tensor tensor, (int left, int right)[] paddingSizes, PaddingMode mode = PaddingMode.CONSTANT, float value = 0.0f)
        {
            if (paddingSizes == null || paddingSizes.Length == 0)
            {
                return Clone(tensor);
            }
            
            int ndim = tensor.NDim;
            int padLen = paddingSizes.Length;
            if (padLen > ndim)
            {
                throw new System.ArgumentException($"Padding length {padLen} exceeds tensor dimensions {ndim}.");
            }

            (int left, int right)[] fullPadding = new (int, int)[ndim];
            int offset = ndim - padLen;
            for (int i = 0; i < ndim; i++)
            {
                if (i < offset)
                {
                    fullPadding[i] = (0, 0);
                }
                else
                {
                    fullPadding[i] = paddingSizes[i - offset];
                }
            }

            switch (mode)
            {
                case PaddingMode.CONSTANT:
                    return ConstantPad(tensor, fullPadding, value);
                case PaddingMode.REFLECT:
                    return ReflectPad(tensor, fullPadding);
                case PaddingMode.REPLICATE:
                    return ReplicatePad(tensor, fullPadding);
                case PaddingMode.CIRCULAR:
                    return CircularPad(tensor, fullPadding);
                default:
                    throw new System.ArgumentException($"{mode} is not supported");
            }
        }

        public static Tensor ConstantPad(Tensor tensor, (int left, int right)[] paddingSizes, float value = 0.0f)
        {
            return PadInternal(tensor, paddingSizes, PaddingMode.CONSTANT, value);
        }

        public static Tensor ReflectPad(Tensor tensor, (int left, int right)[] paddingSizes)
        {
            // Check padding sizes
            for (int i = 0; i < tensor.NDim; i++)
            {
                if (paddingSizes[i].left >= tensor.Size[i] || paddingSizes[i].right >= tensor.Size[i])
                {
                    throw new System.ArgumentException($"Padding size ({paddingSizes[i]}) must be less than the corresponding input dimension size ({tensor.Size[i]}) for REFLECT mode.");
                }
            }
            return PadInternal(tensor, paddingSizes, PaddingMode.REFLECT, 0.0f);
        }

        public static Tensor ReplicatePad(Tensor tensor, (int left, int right)[] paddingSizes)
        {
            return PadInternal(tensor, paddingSizes, PaddingMode.REPLICATE, 0.0f);
        }

        public static Tensor CircularPad(Tensor tensor, (int left, int right)[] paddingSizes)
        {
            return PadInternal(tensor, paddingSizes, PaddingMode.CIRCULAR, 0.0f);
        }

        private static Tensor PadInternal(Tensor tensor, (int left, int right)[] paddingSizes, PaddingMode mode, float constantValue)
        {
            int ndim = tensor.NDim;
            int[] newSize = new int[ndim];
            for (int i = 0; i < ndim; i++)
            {
                newSize[i] = tensor.Size[i] + paddingSizes[i].left + paddingSizes[i].right;
                if (newSize[i] < 0) throw new System.ArgumentException($"Padding results in negative size for dimension {i}");
            }

            Tensor output = new Tensor(new float[Util.ArrayExt.IntProd(newSize)], newSize);
            
            if (mode == PaddingMode.CONSTANT)
            {
                if (constantValue != 0.0f)
                {
                    for(int i=0; i<output.Data.Length; i++) output.Data[i] = constantValue;
                }
            }

            // Recursive copy
            PadRecursive(tensor, output, paddingSizes, mode, 0, 0, 0);

            return output;
        }

        private static void PadRecursive(Tensor src, Tensor dst, (int left, int right)[] pads, PaddingMode mode, int dim, int srcOffset, int dstOffset)
        {
            int ndim = src.NDim;
            int srcDimSize = src.Size[dim];
            int dstDimSize = dst.Size[dim];
            int padLeft = pads[dim].left;
            int padRight = pads[dim].right;
            
            int srcStride = src.Strides[dim];
            int dstStride = dst.Strides[dim];

            if (dim == ndim - 1)
            {
                bool canBlockCopy = (srcStride == 1) && (dstStride == 1);

                if (canBlockCopy)
                {
                    System.Array.Copy(src.Data, srcOffset, dst.Data, dstOffset + padLeft, srcDimSize);
                }
                else
                {
                    for (int i = 0; i < srcDimSize; i++)
                    {
                        dst.Data[dstOffset + (padLeft + i) * dstStride] = src.Data[srcOffset + i * srcStride];
                    }
                }

                // 2. Padding (Left and Right)
                if (mode != PaddingMode.CONSTANT)
                {
                    // Left Padding
                    for (int i = 0; i < padLeft; i++)
                    {
                        int srcIdx = GetPadIndex(i - padLeft, srcDimSize, mode);
                        //Debug.Log($"dst Index: {i} [{dstOffset + i * dstStride}]/{dst.Data.Length}, src Index: {srcIdx} [{srcOffset + srcIdx * srcStride}]/{src.Data.Length}");
                        dst.Data[dstOffset + i * dstStride] = src.Data[srcOffset + srcIdx * srcStride];
                    }
                    for (int i = 0; i < padRight; i++)
                    {
                        int srcIdx = GetPadIndex(srcDimSize + i, srcDimSize, mode);
                        dst.Data[dstOffset + (padLeft + srcDimSize + i) * dstStride] = src.Data[srcOffset + srcIdx * srcStride];
                    }
                }
            }
            else
            {
                for (int i = 0; i < srcDimSize; i++)
                {
                    PadRecursive(src, dst, pads, mode, dim + 1, srcOffset + i * srcStride, dstOffset + (padLeft + i) * dstStride);
                }
                if (mode != PaddingMode.CONSTANT)
                {
                    // Left Padding
                    for (int i = 0; i < padLeft; i++)
                    {
                        int srcIdx = GetPadIndex(i - padLeft, srcDimSize, mode);
                        //Debug.Log($"dst Index: {i} [{dstOffset + i * dstStride}]/{dst.Data.Length}, src Index: {srcIdx} [{srcOffset + srcIdx * srcStride}]/{src.Data.Length}");
                        PadRecursive(src, dst, pads, mode, dim + 1, srcOffset + srcIdx * srcStride, dstOffset + i * dstStride);
                    }

                    // Right Padding
                    for (int i = 0; i < padRight; i++)
                    {
                        int srcIdx = GetPadIndex(srcDimSize + i, srcDimSize, mode);
                        PadRecursive(src, dst, pads, mode, dim + 1, srcOffset + srcIdx * srcStride, dstOffset + (padLeft + srcDimSize + i) * dstStride);
                    }
                }
                else
                {
                }
            }
        }

        private static int GetPadIndex(int index, int size, PaddingMode mode)
        {
            if (index >= 0 && index < size) return index;

            switch (mode)
            {
                case PaddingMode.REFLECT:
                    if (index < 0) return -index;
                    if (index >= size) return 2 * size - 2 - index;
                    return index;

                case PaddingMode.REPLICATE:
                    // Replicate: AAA|ABC|CCC
                    if (index < 0) return 0;
                    if (index >= size) return size - 1;
                    return index;

                case PaddingMode.CIRCULAR:
                    // Circular: C|ABC|A
                    // Modulo arithmetic
                    int mod = index % size;
                    if (mod < 0) mod += size;
                    return mod;
                
                default:
                    return 0;
            }
        }

        public static Tensor PadBackward(Tensor gradOutput, (int left, int right)[] paddingSizes, PaddingMode mode)
        {
            // Calculate original size
            int ndim = gradOutput.NDim;
            int[] originalSize = new int[ndim];
            for (int i = 0; i < ndim; i++)
            {
                originalSize[i] = gradOutput.Size[i] - paddingSizes[i].left - paddingSizes[i].right;
            }
            
            Tensor gradInput = new Tensor(new float[Util.ArrayExt.IntProd(originalSize)], originalSize);
            
            // Recursive accumulation
            PadBackwardRecursive(gradOutput, gradInput, paddingSizes, mode, 0, 0, 0);
            
            return gradInput;
        }

        private static void PadBackwardRecursive(Tensor src, Tensor dst, (int left, int right)[] pads, PaddingMode mode, int dim, int srcOffset, int dstOffset)
        {
            int ndim = src.NDim;
            int dstDimSize = dst.Size[dim]; // Original size
            int srcDimSize = src.Size[dim]; // Padded size
            int padLeft = pads[dim].left;
            int padRight = pads[dim].right;
            
            int srcStride = src.Strides[dim];
            int dstStride = dst.Strides[dim];

            if (dim == ndim - 1)
            {
                // Copy Center
                for (int i = 0; i < dstDimSize; i++)
                {
                    dst.Data[dstOffset + i * dstStride] += src.Data[srcOffset + (padLeft + i) * srcStride];
                }

                if (mode != PaddingMode.CONSTANT)
                {
                    // Accumulate Left Padding
                    for (int i = 0; i < padLeft; i++)
                    {
                        int dstIdx = GetPadIndex(i - padLeft, dstDimSize, mode);
                        dst.Data[dstOffset + dstIdx * dstStride] += src.Data[srcOffset + i * srcStride];
                    }

                    // Accumulate Right Padding
                    for (int i = 0; i < padRight; i++)
                    {
                        int dstIdx = GetPadIndex(dstDimSize + i, dstDimSize, mode);
                        dst.Data[dstOffset + dstIdx * dstStride] += src.Data[srcOffset + (padLeft + dstDimSize + i) * srcStride];
                    }
                }
            }
            else
            {
                // Recursive step
                
                // 1. Center
                for (int i = 0; i < dstDimSize; i++)
                {
                    PadBackwardRecursive(src, dst, pads, mode, dim + 1, srcOffset + (padLeft + i) * srcStride, dstOffset + i * dstStride);
                }

                // 2. Padding
                if (mode != PaddingMode.CONSTANT)
                {
                    // Left Padding
                    for (int i = 0; i < padLeft; i++)
                    {
                        int dstIdx = GetPadIndex(i - padLeft, dstDimSize, mode);
                        // Accumulate src slice [i] into dst slice [dstIdx]
                        PadBackwardRecursive(src, dst, pads, mode, dim + 1, srcOffset + i * srcStride, dstOffset + dstIdx * dstStride);
                    }

                    // Right Padding
                    for (int i = 0; i < padRight; i++)
                    {
                        int dstIdx = GetPadIndex(dstDimSize + i, dstDimSize, mode);
                        PadBackwardRecursive(src, dst, pads, mode, dim + 1, srcOffset + (padLeft + dstDimSize + i) * srcStride, dstOffset + dstIdx * dstStride);
                    }
                }
            }
        }
    }
}