package com.hcifuture.datacollection.inference.filter;

import android.util.Log;

public class ImuFilter {
    private String mode;
    private float fs;
    private float tw;
    private float fc_low;
    private float fc_high;
    private String window_type;
    private float[] h;

    public ImuFilter(String mode, float fs, float tw, float fc_low, float fc_high, String window_type) {
        this.mode = mode;
        this.fs = fs;
        this.tw = tw;
        this.fc_low = fc_low;
        this.fc_high = fc_high;
        this.window_type = window_type;
        initialize_h();
    }

    private void initialize_h() {
        int N;
        float[] n;
        float[] w;
        if (window_type.equals("rec")) {
            N = (((int)(0.91 * fs / tw) + 1) / 2) * 2 + 1;
            n = new float[N];
            w = new float[N];
            for (int i = 0; i < N; i++) {
                n[i] = i - N / 2;
                w[i] = 1;
            }
        } else if (window_type.equals("hanning")) {
            N = (((int)(3.32 * fs / tw) + 1) / 2) * 2 + 1;
            n = new float[N];
            w = new float[N];
            for (int i = 0; i < N; i++) {
                n[i] = i - N / 2;
                w[i] = 0.5f + 0.5f * (float)Math.cos(2.0 * Math.PI / (N - 1) * n[i]);
            }
        } else if (window_type.equals("hamming")) {
            N = (((int)(3.44 * fs / tw) + 1) / 2) * 2 + 1;
            n = new float[N];
            w = new float[N];
            for (int i = 0; i < N; i++) {
                n[i] = i - N / 2;
                w[i] = 0.54f + 0.46f * (float)Math.cos(2.0 * Math.PI / (N - 1) * n[i]);
            }
        } else {
            N = (((int)(5.98 * fs / tw) + 1) / 2) * 2 + 1;
            n = new float[N];
            w = new float[N];
            for (int i = 0; i < N; i++) {
                n[i] = i - N / 2;
                w[i] = 0.42f + 0.5f * (float)Math.cos(2.0 * Math.PI / (N - 1) * n[i])
                       + 0.08f * (float)Math.cos(4 * Math.PI / (N - 1) * n[i]);
            }
        }

        h = new float[N];
        if (mode.equals("low-pass")) {
            float wc = 2.0f * (float)Math.PI * fc_low / fs;
            for (int i = 0; i < N; i++) {
                h[i] = (float)(Math.sin(wc * n[i]) / (Math.PI * n[i]));
            }
            h[N / 2] = wc / (float)Math.PI;
            for (int i = 0; i < N; i++) {
                h[i] *= w[i];
            }
        } else if (mode.equals("high-pass")) {
            float wc = (float)Math.PI - 2 * (float)Math.PI * fc_high / fs;
            for (int i = 0; i < N; i++) {
                h[i] = (float)(Math.sin(wc * n[i]) / (Math.PI * n[i]));
            }
            h[N / 2] = wc / (float)Math.PI;
            for (int i = 0; i < N; i++) {
                h[i] *= w[i] * Math.cos(Math.PI * n[i]);
            }
        } else { // band-pass
            float wc = (float)Math.PI * (fc_high - fc_low) / fs;
            for (int i = 0; i < N; i++) {
                h[i] = (float)(Math.sin(wc * n[i]) / (Math.PI * n[i]));
            }
            h[N / 2] = wc / (float)Math.PI;
            for (int i = 0; i < N; i++) {
                h[i] *= 2 * w[i] * Math.cos(Math.PI * (fc_low + fc_high) / fs * n[i]);
            }
        }
    }

    public float[] filter(float[] input) {
        Log.e("FILTER", input.length + " " + h.length);
        float[] mul = new float[input.length + h.length - 1];
        for (int i = 0; i < input.length + h.length - 1; i++) {
            mul[i] = 0;
        }
        for (int i = 0; i < input.length; i++) {
            for (int j = 0; j < h.length; j++) {
                mul[i + j] += input[i] * h[j];
            }
        }
        float[] result = new float[input.length];
        int bp = (h.length - 1) / 2;
        for (int i = bp; i < bp + input.length; i++) {
            result[i - bp] = mul[i];
        }
        return result;
    }

    public static void unitTest() {
        int T = 2;
        int fs = 100;
        float[] amps = new float[]{48, 32, 24, 16, 12, 10, 8, 6, 4, 3, 2, 1};
        float[] freqs = new float[]{1, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48};
        float[] f = new float[T * fs];
        float[] t = new float[T * fs];
        for (int i = 0; i < T * fs; i++) {
            t[i] = 1.0f * i / fs;
        }
        for (int i = 0; i < amps.length; i++) {
            for (int j = 0; j < T * fs; j++) {
                f[j] += amps[i] * Math.sin(2.0 * Math.PI * freqs[i] * t[j]);
            }
        }
        int tw = 2;
        ImuFilter low_pass = new ImuFilter("low-pass", fs, tw, 7, 0, "hamming");
        ImuFilter high_pass = new ImuFilter("high-pass", fs, tw, 0, 20, "hamming");
        ImuFilter band_pass = new ImuFilter("band-pass", fs, tw, 7, 20, "hamming");

        float[] low_f = low_pass.filter(f);
        float[] high_f = high_pass.filter(f);
        float[] band_f = band_pass.filter(f);

        for (int i = 0; i < 10; i++) {
            Log.e("Filter", "Low " + i + " " + low_f[i]);
        }
        for (int i = 0; i < 10; i++) {
            Log.e("Filter", "High " + i + " " + high_f[i]);
        }
        for (int i = 0; i < 10; i++) {
            Log.e("Filter", "Band " + i + " " + band_f[i]);
        }
    }
}
