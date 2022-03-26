package com.hcifuture.contextactionlibrary.utils.imu;

public class PeakDetector {
    private float _amplitudeMajorPeak = 0.0F;
    private float _amplitudeReference = 0.0F;
    private int _idMajorPeak = -1;
    public float _minNoiseTolerate = 0.0F;
    private float _noiseTolerate;
    private int _windowSize = 0;

    public PeakDetector() {
    }

    public int getIdMajorPeak() {
        return this._idMajorPeak;
    }

    public void setMinNoiseTolerate(float var1) {
        this._minNoiseTolerate = var1;
    }

    public void setWindowSize(int var1) {
        this._windowSize = var1;
    }

    public void update(float var1) {
        int var2 = this._idMajorPeak - 1;
        this._idMajorPeak = var2;
        if (var2 < 0) {
            this._amplitudeMajorPeak = 0.0F;
        }

        float var3 = this._minNoiseTolerate;
        this._noiseTolerate = var3;
        float var4 = this._amplitudeMajorPeak;
        if (var4 / 5.0F > var3) {
            this._noiseTolerate = var4 / 5.0F;
        }

        var4 = this._amplitudeReference - var1;
        var3 = this._noiseTolerate;
        if (var4 < var3) {
            if (var4 < 0.0F && var1 > var3) {
                this._amplitudeReference = var1;
                if (var1 > this._amplitudeMajorPeak) {
                    this._idMajorPeak = this._windowSize - 1;
                    this._amplitudeMajorPeak = var1;
                }
            }
        } else {
            this._amplitudeReference = var1;
        }

    }
}
