package com.example.contextactionlibrary.utils;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;

public class EventIMURT {
    protected ArrayList<Float> _fv = new ArrayList();
    protected boolean _gotAcc = false;
    protected boolean _gotGyro = false;
    protected Highpass3C _highpassAcc = new Highpass3C();
    protected Highpass3C _highpassGyro = new Highpass3C();
    protected Lowpass3C _lowpassAcc = new Lowpass3C();
    protected Lowpass3C _lowpassGyro = new Lowpass3C();
    protected int _numberFeature;
    protected Resample3C _resampleAcc = new Resample3C();
    protected Resample3C _resampleGyro = new Resample3C();
    protected int _sizeFeatureWindow;
    protected long _sizeWindowNs;
    protected Slope3C _slopeAcc = new Slope3C();
    protected Slope3C _slopeGyro = new Slope3C();
    protected long _syncTime = 0L;
    protected Deque<Float> _xsAcc = new ArrayDeque();
    protected Deque<Float> _xsGyro = new ArrayDeque();
    protected Deque<Float> _ysAcc = new ArrayDeque();
    protected Deque<Float> _ysGyro = new ArrayDeque();
    protected Deque<Float> _zsAcc = new ArrayDeque();
    protected Deque<Float> _zsGyro = new ArrayDeque();

    public EventIMURT() {
    }

    public void processGyro() {
        Point3f var1 = this._resampleGyro.getResults().point;
        float var2 = 2500000.0F / (float)this._resampleGyro.getInterval();
        var1 = this._slopeGyro.update(var1, var2);
        var1 = this._lowpassGyro.update(var1);
        var1 = this._highpassGyro.update(var1);
        this._xsGyro.add(var1.x);
        this._ysGyro.add(var1.y);
        this._zsGyro.add(var1.z);
        long var3 = this._resampleGyro.getInterval();
        int var5 = (int)(this._sizeWindowNs / var3);

        while(this._xsGyro.size() > var5) {
            this._xsGyro.removeFirst();
            this._ysGyro.removeFirst();
            this._zsGyro.removeFirst();
        }

    }

    public void reset() {
        this._xsAcc.clear();
        this._ysAcc.clear();
        this._zsAcc.clear();
        this._xsGyro.clear();
        this._ysGyro.clear();
        this._zsGyro.clear();
        this._gotAcc = false;
        this._gotGyro = false;
        this._syncTime = 0L;
    }

    public ArrayList<Float> scaleGyroData(ArrayList<Float> var1, float var2) {
        for(int var3 = var1.size() / 2; var3 < var1.size(); ++var3) {
            var1.set(var3, (Float)var1.get(var3) * var2);
        }

        return var1;
    }
}
