namespace SimpleNN.Graph
{
    using System.IO;
    using SimpleNN.Tensor;
    
    public partial class TensorBox
    {
        public static byte[] ToBinary(TensorBox box)
        {
            using var ms = new MemoryStream();
            using (var writer = new BinaryWriter(ms))
            {
                // requireGrad (byte)
                writer.Write(box.RequireGrad ? (byte)1 : (byte)0);

                // Tensor (binary)
                byte[] tensorData = Tensor.ToBinary(box.GetTensor());
                writer.Write(tensorData);
            }
            return ms.ToArray();
        }

        public static TensorBox FromBinary(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var reader = new BinaryReader(ms);
            return FromBinary(reader);
        }

        public static TensorBox FromBinary(BinaryReader reader)
        {
            // requireGrad (byte)
            bool requireGrad = reader.ReadByte() != 0;

            // Tensor (binary)
            // TensorConverter.FromBinary(reader) assumes it reads from current position
            Tensor tensor = Tensor.FromBinary(reader);

            return new TensorBox(tensor, requireGrad);
        }
    }
}