package com.example.contextactionlibrary.data;

import com.example.contextactionlibrary.utils.Highpass1C;
import com.example.contextactionlibrary.utils.Highpass3C;
import com.example.contextactionlibrary.utils.Lowpass1C;
import com.example.contextactionlibrary.utils.Lowpass3C;
import com.example.contextactionlibrary.utils.PeakDetector;
import com.example.contextactionlibrary.utils.Point3f;
import com.example.contextactionlibrary.utils.Resample3C;
import com.example.contextactionlibrary.utils.Sample3C;
import com.example.contextactionlibrary.utils.Slope3C;

import java.util.ArrayList;
import java.util.List;

public class Preprocess {

    private long SAMPLINGINTERVALNS = 2500000L;
    private long WINDOW_NS = 160000000L;

    private boolean gotAcc = false;
    private boolean gotGyro = false;
    protected Highpass3C highpassAcc = new Highpass3C();
    protected Highpass3C highpassGyro = new Highpass3C();
    protected Lowpass3C lowpassAcc = new Lowpass3C();
    protected Lowpass3C lowpassGyro = new Lowpass3C();
    private Resample3C resampleAcc = new Resample3C();
    private Resample3C resampleGyro = new Resample3C();
    protected Slope3C slopeAcc = new Slope3C();
    protected Slope3C slopeGyro = new Slope3C();
    private long syncTime = 0L;
    private List<Float> xsAcc = new ArrayList<>();
    private List<Float> ysAcc = new ArrayList<>();
    private List<Float> zsAcc = new ArrayList<>();
    private List<Float> xsGyro = new ArrayList<>();
    private List<Float> ysGyro = new ArrayList<>();
    private List<Float> zsGyro = new ArrayList<>();
    private List<Long> timestamps = new ArrayList<>();

    private Highpass1C highpassKey = new Highpass1C();
    private Lowpass1C lowpassKey = new Lowpass1C();
    private PeakDetector peakDetectorPositive = new PeakDetector();
    private boolean wasPeakApproaching = true;

    private static int proxThreshold = 4;
    private long lastNearTime = 0L;
    private long lastFarTime = 0L;

    public List<Float> getXsAcc() {
        return xsAcc;
    }
    public List<Float> getYsAcc() {
        return ysAcc;
    }
    public List<Float> getZsAcc() {
        return zsAcc;
    }
    public List<Float> getXsGyro() {
        return xsGyro;
    }
    public List<Float> getYsGyro() {
        return ysGyro;
    }
    public List<Float> getZsGyro() {
        return zsGyro;
    }
    public List<Long> getTimestamps() {
        return timestamps;
    }

    private static Preprocess instance;

    public static Preprocess getInstance() {
        if (instance == null)
            instance = new Preprocess();
        return instance;
    }

    public Preprocess() {
        lowpassAcc.setPara(1.0F);
        lowpassGyro.setPara(1.0F);
        highpassAcc.setPara(0.05F);
        highpassGyro.setPara(0.05F);
        lowpassKey.setPara(0.2F);
        highpassKey.setPara(0.2F);
        peakDetectorPositive.setMinNoiseTolerate(0.05f);
        peakDetectorPositive.setWindowSize(64);
    }

    public void processAccAndKeySignal() {
        Sample3C sample = resampleAcc.getResults();
        Point3f point1 = slopeAcc.update(sample.point, 2500000.0F / (float)resampleAcc.getInterval());
        Point3f point3 = highpassAcc.update(lowpassAcc.update(point1));
        xsAcc.add(point3.x);
        ysAcc.add(point3.y);
        zsAcc.add(point3.z);
        timestamps.add(sample.t);
        int size = (int)(WINDOW_NS / resampleAcc.getInterval());

        while(xsAcc.size() > size) {
            xsAcc.remove(0);
            ysAcc.remove(0);
            zsAcc.remove(0);
            timestamps.remove(0);
        }

        peakDetectorPositive.update(highpassKey.update(lowpassKey.update(point1.z)));
    }

    public void processGyro() {
        Point3f point = resampleGyro.getResults().point;
        point = highpassGyro.update(lowpassGyro.update(slopeGyro.update(point, 2500000.0F / (float)resampleGyro.getInterval())));
        xsGyro.add(point.x);
        ysGyro.add(point.y);
        zsGyro.add(point.z);
        int size = (int)(WINDOW_NS / resampleGyro.getInterval());

        while(xsGyro.size() > size) {
            xsGyro.remove(0);
            ysGyro.remove(0);
            zsGyro.remove(0);
        }
    }

    public int[] shouldRunTapModel(int seqLength) {
        int diff = (int)((resampleAcc.getResults().t - resampleGyro.getResults().t) / resampleAcc.getInterval());
        int peakIdx = peakDetectorPositive.getIdMajorPeak();
        if (peakIdx > 20)
            wasPeakApproaching = true;
        int accIdx = peakIdx - 6;
        int gyroIdx = accIdx - diff;
        if (accIdx >= 0 && gyroIdx >= 0) {
            if (accIdx + seqLength < zsAcc.size() && gyroIdx + seqLength < zsAcc.size() && wasPeakApproaching && peakIdx <= 20) {
                wasPeakApproaching = false;
                return new int[]{accIdx, gyroIdx};
            }
        }
        return new int[]{-1, -1};
    }

    public void preprocessIMU(int type, float x, float y, float z, long timestamp) {
        if (type == 1) {
            gotAcc = true;
            if (0L == syncTime)
                resampleAcc.init(x, y, z, timestamp, SAMPLINGINTERVALNS);
            if (!gotGyro)
                return;
        } else if (type == 4) {
            gotGyro = true;
            if (0L == syncTime)
                resampleGyro.init(x, y, z, timestamp, SAMPLINGINTERVALNS);
            if (!gotAcc)
                return;
        }
        if (0L == syncTime) {
            syncTime = timestamp;
            resampleAcc.setSyncTime(timestamp);
            resampleGyro.setSyncTime(syncTime);
            slopeAcc.init(resampleAcc.getResults().point);
            slopeGyro.init(resampleGyro.getResults().point);
            lowpassAcc.init(new Point3f(0.0F, 0.0F, 0.0F));
            lowpassGyro.init(new Point3f(0.0F, 0.0F, 0.0F));
            highpassAcc.init(new Point3f(0.0F, 0.0F, 0.0F));
            highpassGyro.init(new Point3f(0.0F, 0.0F, 0.0F));
            lowpassKey.init(0.0F);
            highpassKey.init(0.0F);
        } else {
            if (type == 1)
                while(resampleAcc.update(x, y, z, timestamp))
                    processAccAndKeySignal();
            else if (type == 4)
                while(resampleGyro.update(x, y, z, timestamp))
                    processGyro();
        }
    }

    public void preprocessProx(float prox, long timestamp) {
        if ((int)prox < proxThreshold)
            lastNearTime = timestamp;
        else
            lastFarTime = timestamp;
    }

    public long checkLastNear(long threshold, long lastRecognized) {
        long lastTimestamp = timestamps.get(timestamps.size() - 1);
        if (lastTimestamp - lastNearTime < threshold && lastTimestamp - lastRecognized > 5 * 1e9)
            return lastTimestamp;
        return -1;
    }

    public void reset() {
        gotAcc = false;
        gotGyro = false;
        syncTime = 0L;
        xsAcc.clear();
        ysAcc.clear();
        zsAcc.clear();
        xsGyro.clear();
        ysGyro.clear();
        zsGyro.clear();
        timestamps.clear();
    }
}
