namespace SimpleNN.Graph
{
    using SimpleNN.Graph.Functions;
    public partial class TensorBox
    {
        private static Context Reshape(Context ctx,int[] shape) {
            return ReshapeFunction.Forward(new(){ newSize = shape }, ctx);
        }
        public TensorBox Reshape(params int[] shape)
        {
            return new(Reshape(_ctx, shape));
        }
        /// <summary>
        /// 2つの次元を入れ替えた新しいTensorBoxを返す.
        /// </summary>
        private static Context Transpose(Context ctx, int dim0, int dim1)
        {
            return TransposeFunction.Forward(new(){ dim0 = dim0, dim1 = dim1 }, ctx);
        }

        public TensorBox Transpose(int dim0, int dim1)
        {
            return new TensorBox(Transpose(_ctx, dim0, dim1));
        }

        /// <summary>
        /// 指定した位置に次元(サイズ1)を挿入した新しいTensorBoxを返す.
        /// </summary>
        private static Context Unsqueeze(Context ctx, int dim)
        {
            return UnsqueezeFunction.Forward(new(){ dim = dim }, ctx);
        }

        public TensorBox Unsqueeze(int dim)
        {
            return new TensorBox(Unsqueeze(_ctx, dim));
        }

        private static Context Squeeze(Context ctx, int dim)
        {
            return SqueezeFunction.Forward(new(){ dim = dim }, ctx);
        }
        public TensorBox Squeeze(int dim)
        {
            return new TensorBox(Squeeze(_ctx, dim));
        }
        private static Context Squeeze(Context ctx)
        {
            return AllSqueezeFunction.Forward(ctx);
        }
        public TensorBox Squeeze()
        {
            return new TensorBox(Squeeze(_ctx));
        }
        private static Context Flatten(Context ctx, int startIndex = 0, int endIndex = -1) {
            return FlattenFunction.Forward(new(){ startIndex = startIndex, endIndex = endIndex }, ctx);
        }
        public TensorBox Flatten(int startIndex = 0, int endIndex = -1)
        {
            return new(Flatten(_ctx, startIndex, endIndex));
        }
    }
}