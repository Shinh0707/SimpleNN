using System;
using UnityEngine;
using SimpleNN.Util;

namespace SimpleNN.Tensor
{
    public partial class Tensor
    {
        public static Tensor AsRGB(Texture texture)
        {
            return Convert(texture, (Color c, out float v1, out float v2, out float v3) =>
            {
                v1 = c.r;
                v2 = c.g;
                v3 = c.b;
            });
        }

        public void SetAsRGB(Texture texture)
        {
            Convert(texture, _data, (Color c, out float v1, out float v2, out float v3) =>
            {
                v1 = c.r;
                v2 = c.g;
                v3 = c.b;
            });
        }

        public static Tensor AsRGBA(Texture texture)
        {
            return Convert(texture, (Color c, out float v1, out float v2, out float v3, out float v4) =>
            {
                v1 = c.r;
                v2 = c.g;
                v3 = c.b;
                v4 = c.a;
            });
        }

        public void SetAsRGBA(Texture texture)
        {
            Convert(texture, _data, (Color c, out float v1, out float v2, out float v3, out float v4) =>
            {
                v1 = c.r;
                v2 = c.g;
                v3 = c.b;
                v4 = c.a;
            });
        }

        public static Tensor AsHSV(Texture texture)
        {
            return Convert(texture, (Color c, out float v1, out float v2, out float v3) =>
            {
                ColorUtil.RGBToHSV(c.r, c.g, c.b, out v1, out v2, out v3);
            });
        }

        public void SetAsHSV(Texture texture)
        {
            Convert(texture, _data, (Color c, out float v1, out float v2, out float v3) =>
            {
                ColorUtil.RGBToHSV(c.r, c.g, c.b, out v1, out v2, out v3);
            });
        }

        public static Tensor AsConicalHSV(Texture texture)
        {
            return Convert(texture, (Color c, out float v1, out float v2, out float v3) =>
            {
                ColorUtil.RGBToHSV(c.r, c.g, c.b, out float h, out float s, out float v);
                ColorUtil.HSVToConical(h, s, v, out v1, out v2, out v3);
            });
        }

        public void SetAsConicalHSV(Texture texture)
        {
            Convert(texture, _data, (Color c, out float v1, out float v2, out float v3) =>
            {
                ColorUtil.RGBToHSV(c.r, c.g, c.b, out float h, out float s, out float v);
                ColorUtil.HSVToConical(h, s, v, out v1, out v2, out v3);
            });
        }

        private delegate void PixelProcessor(Color c, out float v1, out float v2, out float v3);

        private static Tensor Convert(Texture texture, PixelProcessor processor)
        {
            if (texture is Texture2D tex2D)
            {
                return Convert(tex2D, processor);
            }
            else if (texture is WebCamTexture webCamTex)
            {
                return Convert(webCamTex, processor);
            }
            
            throw new NotSupportedException($"Texture type {texture.GetType().Name} is not supported. Only Texture2D and WebCamTexture are supported.");
        }

        private static Tensor Convert(Texture2D texture, PixelProcessor processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;

            float[] data = new float[3 * pixelCount];
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
            }

            return new Tensor(data, new int[] { 3, height, width });
        }

        private static Tensor Convert(WebCamTexture texture, PixelProcessor processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;

            float[] data = new float[3 * pixelCount];
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
            }

            return new Tensor(data, new int[] { 3, height, width });
        }

        private static void Convert(Texture texture, float[] data, PixelProcessor processor)
        {
            if (texture is Texture2D tex2D)
            {
                Convert(tex2D, data, processor);
            }
            else if (texture is WebCamTexture webCamTex)
            {
                Convert(webCamTex, data, processor);
            }
            else
            {
                throw new NotSupportedException($"Texture type {texture.GetType().Name} is not supported. Only Texture2D and WebCamTexture are supported.");
            }
        }

        private static void Convert(Texture2D texture, float[] data, PixelProcessor processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
            }
        }

        private static void Convert(WebCamTexture texture, float[] data, PixelProcessor processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
            }
        }

        private delegate void PixelProcessor4(Color c, out float v1, out float v2, out float v3, out float v4);

        private static Tensor Convert(Texture texture, PixelProcessor4 processor)
        {
            if (texture is Texture2D tex2D)
            {
                return Convert(tex2D, processor);
            }
            else if (texture is WebCamTexture webCamTex)
            {
                return Convert(webCamTex, processor);
            }

            throw new NotSupportedException($"Texture type {texture.GetType().Name} is not supported. Only Texture2D and WebCamTexture are supported.");
        }

        private static Tensor Convert(Texture2D texture, PixelProcessor4 processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;

            float[] data = new float[4 * pixelCount];
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3, out float v4);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
                data[i + 3 * channelStride] = v4;
            }

            return new Tensor(data, new int[] { 4, height, width });
        }

        private static Tensor Convert(WebCamTexture texture, PixelProcessor4 processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;

            float[] data = new float[4 * pixelCount];
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3, out float v4);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
                data[i + 3 * channelStride] = v4;
            }

            return new Tensor(data, new int[] { 4, height, width });
        }

        private static void Convert(Texture texture, float[] data, PixelProcessor4 processor)
        {
            if (texture is Texture2D tex2D)
            {
                Convert(tex2D, data, processor);
            }
            else if (texture is WebCamTexture webCamTex)
            {
                Convert(webCamTex, data, processor);
            }
            else
            {
                throw new NotSupportedException($"Texture type {texture.GetType().Name} is not supported. Only Texture2D and WebCamTexture are supported.");
            }
        }

        private static void Convert(Texture2D texture, float[] data, PixelProcessor4 processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3, out float v4);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
                data[i + 3 * channelStride] = v4;
            }
        }

        private static void Convert(WebCamTexture texture, float[] data, PixelProcessor4 processor)
        {
            int width = texture.width;
            int height = texture.height;
            int pixelCount = width * height;
            int channelStride = pixelCount;
            Color[] pixels = texture.GetPixels();

            for (int i = 0; i < pixelCount; i++)
            {
                processor(pixels[i], out float v1, out float v2, out float v3, out float v4);
                data[i] = v1;
                data[i + channelStride] = v2;
                data[i + 2 * channelStride] = v3;
                data[i + 3 * channelStride] = v4;
            }
        }
    }
}
