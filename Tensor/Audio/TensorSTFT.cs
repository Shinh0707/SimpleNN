namespace SimpleNN.Tensor.Audio
{
    using System;
    using System.Numerics;
    using SimpleNN.Tensor;

    public static class TensorFFT
    {
        public static (Tensor, Tensor) STFT(Tensor input, int n_fft, int hop_length = -1, int win_length = -1, Tensor window = null, bool center = true, PaddingMode mode = PaddingMode.REFLECT, bool normalized = false, bool onesided = true, FFTPlan plan = null)
        {
            if (hop_length <= 0) hop_length = n_fft / 4;
            if (win_length <= 0) win_length = n_fft;
            if (plan == null || plan.N != n_fft)
            {
                plan = new FFTPlan(n_fft);
            }

            // 1. Padding
            Tensor paddedInput = input;
            if ((input.NDim == 2) && (input.Size[0] == 1)) paddedInput = Tensor.Squeeze(input, 0);
            else if ((input.NDim == 2) && (input.Size[1] == 1)) paddedInput = Tensor.Squeeze(input, 1);

            if (center)
            {
                int pad = n_fft / 2;
                if (paddedInput.NDim == 1) paddedInput = Tensor.Pad(paddedInput, new (int, int)[] { (pad, pad) }, mode, 0.0f);
                else throw new NotImplementedException("Only 1D input is supported for STFT padding.");
            }

            float[] windowData = (window is not null) ? window.Data : GetHannWindow(win_length);

            int inputLength = paddedInput.Size[paddedInput.NDim - 1];
            int n_frames = (inputLength - n_fft) / hop_length + 1;
            int n_freq = onesided ? (n_fft / 2 + 1) : n_fft;

            float[] outReal = new float[n_freq * n_frames];
            float[] outImag = new float[n_freq * n_frames];
            
            float[] paddedInputData = paddedInput.Data;
            
            float[] frameBuffer = new float[n_fft];
            float[] realBuffer = new float[n_fft];
            float[] imagBuffer = new float[n_fft];
            
            int vectorSize = Vector<float>.Count;
            for (int t = 0; t < n_frames; t++)
            {
                int start = t * hop_length;
                
                // --- Window Apply ---
                int i = 0;
                if (start + n_fft <= inputLength)
                {
                    for (; i <= n_fft - vectorSize; i += vectorSize)
                    {
                        Vector<float> vInput = new Vector<float>(paddedInputData, start + i);
                        Vector<float> vWindow;
                        
                        if (i + vectorSize <= win_length) vWindow = new Vector<float>(windowData, i);
                        else if (i >= win_length) vWindow = Vector<float>.Zero;
                        else
                        {
                            float[] tempW = new float[vectorSize];
                            for (int k = 0; k < vectorSize; k++) tempW[k] = (i + k < win_length) ? windowData[i + k] : 0.0f;
                            vWindow = new Vector<float>(tempW);
                        }
                        (vInput * vWindow).CopyTo(frameBuffer, i);
                    }
                }
                for (; i < n_fft; i++)
                {
                    if (start + i < inputLength)
                    {
                        float w = (i < win_length) ? windowData[i] : 0.0f;
                        frameBuffer[i] = paddedInputData[start + i] * w;
                    }
                    else frameBuffer[i] = 0;
                }

                FFT(frameBuffer, realBuffer, imagBuffer, plan);
                for (int f = 0; f < n_freq; f++)
                {
                    outReal[f * n_frames + t] = realBuffer[f];
                    outImag[f * n_frames + t] = imagBuffer[f];
                }
            }

            if (normalized)
            {
                float normFactor = 1.0f / (float)Math.Sqrt(n_fft);
                ApplyNormalization(outReal, normFactor);
                ApplyNormalization(outImag, normFactor);
            }

            int[] shape = new int[] { n_freq, n_frames };
            return (new Tensor(outReal, shape), new Tensor(outImag, shape));
        }

        private static void ApplyNormalization(float[] data, float factor)
        {
            int i = 0;
            int len = data.Length;
            int vectorSize = Vector<float>.Count;
            Vector<float> vFactor = new Vector<float>(factor);
            for (; i <= len - vectorSize; i += vectorSize)
            {
                (new Vector<float>(data, i) * vFactor).CopyTo(data, i);
            }
            for (; i < len; i++) data[i] *= factor;
        }

        private static float[] GetHannWindow(int length)
        {
            float[] window = new float[length];
            for (int i = 0; i < length; i++) window[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / length));
            return window;
        }

        // Simple Radix-2 FFT (Cooley-Tukey)
        // input must be power of 2 length
        // realOut and imagOut must be same length as input
        private static void FFT(float[] input, float[] realOut, float[] imagOut, FFTPlan plan)
        {
            int n = plan.N;
            int[] bitReverse = plan.BitReverseTable;
            
            // 1. Bit Reversal (計算なし、配列コピーのみ)
            for (int i = 0; i < n; i++)
            {
                int j = bitReverse[i];
                realOut[j] = input[i];
                imagOut[j] = 0;
            }

            // 2. Butterfly Operation
            // Cooley-Tukey with Precomputed Twiddle Factors
            int m = (int)Math.Log(n, 2);
            
            for (int s = 1; s <= m; s++)
            {
                int m2 = 1 << s;       // 2, 4, 8, ...
                int m1 = m2 >> 1;      // 1, 2, 4, ...
                
                // 事前計算テーブルへのストライド
                // k番目の回転因子 W_N^k は、テーブルの k * (N/m2) 番目に対応
                int stride = n / m2;

                for (int k = 0; k < n; k += m2)
                {
                    for (int j = 0; j < m1; j++)
                    {
                        // テーブル参照 (計算コストほぼゼロ)
                        int tblIdx = j * stride;
                        float w_r = plan.CosTable[tblIdx];
                        float w_i = plan.SinTable[tblIdx];

                        int t_idx = k + j + m1;
                        int u_idx = k + j;

                        // 複素乗算
                        float t_r = w_r * realOut[t_idx] - w_i * imagOut[t_idx];
                        float t_i = w_r * imagOut[t_idx] + w_i * realOut[t_idx];

                        float u_r = realOut[u_idx];
                        float u_i = imagOut[u_idx];

                        // バタフライ加減算
                        realOut[u_idx] = u_r + t_r;
                        imagOut[u_idx] = u_i + t_i;

                        realOut[t_idx] = u_r - t_r;
                        imagOut[t_idx] = u_i - t_i;
                    }
                }
            }
        }
    }
}

