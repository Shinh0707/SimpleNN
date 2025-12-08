namespace SimpleNN.Tensor
{
    using System;
    using System.IO;

    public partial class Tensor
    {
        public static byte[] ToBinary(Tensor tensor)
        {
            // If the underlying data length doesn't match TotalSize, it implies a slice or shared buffer.
            // In this case, we must make it contiguous to safely serialize exactly TotalSize elements.
            // This will reset strides to default C-strides.
            if (tensor.Data.Length != (int)tensor.TotalSize)
            {
                tensor = Tensor.Contiguous(tensor);
            }

            using var ms = new MemoryStream();
            using (var writer = new BinaryWriter(ms))
            {
                // ndim (byte)
                writer.Write((byte)tensor.NDim);

                // dims (ushort * ndim)
                for (int i = 0; i < tensor.NDim; i++)
                {
                    writer.Write((ushort)tensor.Size[i]);
                }

                // strides (ushort * ndim)
                for (int i = 0; i < tensor.NDim; i++)
                {
                    writer.Write((ushort)tensor.Strides[i]);
                }

                // totalSize (ulong)
                writer.Write((ulong)tensor.TotalSize);

                // data (float32 * totalSize)
                float[] data = tensor.Data;
                if (data.Length != (int)tensor.TotalSize)
                {
                    // Should not happen due to Contiguous() check above, but safety check
                    throw new InvalidOperationException("Tensor data length mismatch during serialization.");
                }

                byte[] byteData = new byte[data.Length * sizeof(float)];
                Buffer.BlockCopy(data, 0, byteData, 0, byteData.Length);
                writer.Write(byteData);
            }
            return ms.ToArray();
        }

        public static Tensor FromBinary(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);
            return FromBinary(reader);
        }

        public static Tensor FromBinary(BinaryReader reader)
        {
            // ndim (byte)
            byte ndim = reader.ReadByte();

            // dims (ushort * ndim)
            int[] size = new int[ndim];
            for (int i = 0; i < ndim; i++)
            {
                size[i] = (int)reader.ReadUInt16();
            }

            // strides (ushort * ndim)
            int[] strides = new int[ndim];
            for (int i = 0; i < ndim; i++)
            {
                strides[i] = (int)reader.ReadUInt16();
            }

            // totalSize (ulong)
            ulong totalSize = reader.ReadUInt64();

            // data (float32 * totalSize)
            // We need to read totalSize floats.
            // Since totalSize can be large, we should be careful.
            // But arrays are int-indexed in C#.
            if (totalSize > int.MaxValue)
            {
                throw new Exception("Tensor too large for C# array");
            }

            int count = (int)totalSize;
            float[] data = new float[count];
            
            // Read floats
            // Optimization: Read bytes and convert? Or loop ReadSingle?
            // BinaryReader.ReadSingle() in a loop is slow.
            // Better to read bytes and BlockCopy.
            byte[] byteData = reader.ReadBytes(count * sizeof(float));
            Buffer.BlockCopy(byteData, 0, data, 0, byteData.Length);

            return new Tensor(data, size, strides);
        }
    }
}
