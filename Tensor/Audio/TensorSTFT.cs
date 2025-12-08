namespace SimpleNN.Tensor.Audio
{
    using System;
    using System.Numerics;
    using SimpleNN.Tensor;
    using SimpleNN.Util;

    public static class TensorFFT
    {
        public static Tensor STFT(Tensor input, int n_fft, int hop_length = -1, int win_length = -1, Tensor window = null, bool center = true, PaddingMode mode = PaddingMode.REFLECT, bool normalized = false, bool onesided = true, bool return_complex = false)
        {
            if (hop_length <= 0) hop_length = n_fft / 4;
            if (win_length <= 0) win_length = n_fft;

            // 1. Padding
            Tensor paddedInput = input;
            if ((input.NDim == 2) && (input.Size[0] == 1))
            {
                paddedInput = Tensor.Squeeze(input, 0);
            }
            else if ((input.NDim == 2) && (input.Size[1] == 1))
            {
                paddedInput = Tensor.Squeeze(input, 1);
            }
            if (center)
            {
                int pad = n_fft / 2;
                // Only 1D padding supported in this context for now
                if (paddedInput.NDim == 1)
                {
                    paddedInput = Tensor.Pad(paddedInput, new (int, int)[] { (pad, pad) }, mode, 0.0f);
                }
                else
                {
                    throw new NotImplementedException("Only 1D input is supported for STFT padding.");
                }
            }

            // 2. Windowing
            float[] windowData;
            if (window is not null)
            {
                if (window.Size[0] != win_length)
                {
                    throw new ArgumentException($"Window size {window.Size[0]} must match win_length {win_length}");
                }
                windowData = window.Data;
            }
            else
            {
                windowData = GetHannWindow(win_length);
            }

            // 3. Framing
            int inputLength = paddedInput.Size[paddedInput.NDim - 1]; // Assuming last dim is time
            int n_frames = (inputLength - n_fft) / hop_length + 1;
            
            // Output shape: (..., n_fft/2 + 1, n_frames, 2) if onesided and return_complex=false (real/imag)
            // Or (..., n_fft, n_frames, 2) if not onesided
            // For now, let's assume 1D input for simplicity, or treat last dim as time
            
            // We will return (frequency, time, 2) where 2 is real/imag
            int n_freq = onesided ? (n_fft / 2 + 1) : n_fft;
            
            // Prepare output data
            // Batch support is tricky without more advanced tensor ops, assuming 1D input for now
            if (input.NDim != 1)
            {
                throw new NotImplementedException("Only 1D input is supported for now.");
            }

            float[] outputData = new float[n_freq * n_frames * 2];
            
            float[] frameBuffer = new float[n_fft];
            float[] realBuffer = new float[n_fft];
            float[] imagBuffer = new float[n_fft];

            int vectorSize = Vector<float>.Count;

            for (int t = 0; t < n_frames; t++)
            {
                int start = t * hop_length;
                
                // Copy frame and apply window
                // Optimized with Vector<float>
                int i = 0;
                
                // We can only vectorize if we are sure we are within bounds of paddedInput
                // paddedInput length is inputLength
                // We are accessing [start + i]
                // Max index is start + n_fft - 1
                // If start + n_fft <= inputLength, we are safe to read contiguously without bounds check per element
                // (except for the last vector which might need masking, but we can just handle tail separately)
                
                if (start + n_fft <= inputLength)
                {
                    // Fast path
                    for (; i <= n_fft - vectorSize; i += vectorSize)
                    {
                        Vector<float> vInput = new Vector<float>(paddedInput.Data, start + i);
                        
                        // Window might be shorter than n_fft (padded with 0)
                        // But usually win_length <= n_fft.
                        // If i >= win_length, window is 0.
                        
                        Vector<float> vWindow;
                        if (i + vectorSize <= win_length)
                        {
                            vWindow = new Vector<float>(windowData, i);
                        }
                        else if (i >= win_length)
                        {
                            vWindow = Vector<float>.Zero;
                        }
                        else
                        {
                            // Partial window overlap
                            float[] tempW = new float[vectorSize];
                            for (int k = 0; k < vectorSize; k++)
                            {
                                tempW[k] = (i + k < win_length) ? windowData[i + k] : 0.0f;
                            }
                            vWindow = new Vector<float>(tempW);
                        }

                        Vector<float> vRes = vInput * vWindow;
                        vRes.CopyTo(frameBuffer, i);
                    }
                }
                
                // Handle remaining or slow path
                for (; i < n_fft; i++)
                {
                    if (start + i < inputLength)
                    {
                        float val = paddedInput.Data[start + i];
                        float w = (i < win_length) ? windowData[i] : 0.0f;
                        frameBuffer[i] = val * w;
                    }
                    else
                    {
                        frameBuffer[i] = 0;
                    }
                }

                // FFT
                FFT(frameBuffer, realBuffer, imagBuffer);

                // Copy to output
                for (int f = 0; f < n_freq; f++)
                {
                    // (Freq, Time, Real/Imag)
                    outputData[(f * n_frames + t) * 2 + 0] = realBuffer[f];
                    outputData[(f * n_frames + t) * 2 + 1] = imagBuffer[f];
                }
            }

            if (normalized)
            {
                float normFactor = 1.0f / (float)Math.Sqrt(n_fft);
                int len = outputData.Length;
                int i = 0;
                Vector<float> vNorm = new Vector<float>(normFactor);
                
                for (; i <= len - vectorSize; i += vectorSize)
                {
                    Vector<float> vOut = new Vector<float>(outputData, i);
                    vOut *= vNorm;
                    vOut.CopyTo(outputData, i);
                }
                
                for (; i < len; i++)
                {
                    outputData[i] *= normFactor;
                }
            }

            return new Tensor(outputData, new int[] { n_freq, n_frames, 2 });
        }

        private static float[] GetHannWindow(int length)
        {
            float[] window = new float[length];
            for (int i = 0; i < length; i++)
            {
                window[i] = 0.5f * (1.0f - MathF.Cos(2.0f * MathF.PI * i / length));
            }
            return window;
        }

        // Simple Radix-2 FFT (Cooley-Tukey)
        // input must be power of 2 length
        // realOut and imagOut must be same length as input
        private static void FFT(float[] input, float[] realOut, float[] imagOut)
        {
            int n = input.Length;
            int m = (int)Math.Log(n, 2);
            
            if ((1 << m) != n)
            {
                throw new ArgumentException("n_fft must be power of 2 for this simple FFT implementation.");
            }

            // Bit reversal
            for (int i = 0; i < n; i++)
            {
                int j = 0;
                int k = i;
                for (int l = 0; l < m; l++)
                {
                    j = (j << 1) | (k & 1);
                    k >>= 1;
                }
                realOut[j] = input[i];
                imagOut[j] = 0;
            }

            // Butterfly
            float mtpi = -2.0f * MathF.PI;
            for (int s = 1; s <= m; s++)
            {
                int m2 = 1 << s;
                int m1 = m2 >> 1;
                float theta = mtpi / m2;
                float w_m_r = MathF.Cos(theta);
                float w_m_i = MathF.Sin(theta);

                for (int k = 0; k < n; k += m2)
                {
                    float w_r = 1.0f;
                    float w_i = 0.0f;
                    for (int j = 0; j < m1; j++)
                    {
                        int t_idx = k + j + m1;
                        int u_idx = k + j;

                        float t_r = w_r * realOut[t_idx] - w_i * imagOut[t_idx];
                        float t_i = w_r * imagOut[t_idx] + w_i * realOut[t_idx];

                        float u_r = realOut[u_idx];
                        float u_i = imagOut[u_idx];

                        realOut[u_idx] = u_r + t_r;
                        imagOut[u_idx] = u_i + t_i;

                        realOut[t_idx] = u_r - t_r;
                        imagOut[t_idx] = u_i - t_i;

                        float tmp_r = w_r * w_m_r - w_i * w_m_i;
                        w_i = w_r * w_m_i + w_i * w_m_r;
                        w_r = tmp_r;
                    }
                }
            }
        }
    }
}

