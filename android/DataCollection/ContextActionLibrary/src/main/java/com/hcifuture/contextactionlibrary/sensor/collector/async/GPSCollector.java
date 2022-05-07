package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.annotation.SuppressLint;
import android.content.Context;
import android.location.GnssStatus;
import android.location.GpsSatellite;
import android.location.GpsStatus;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.GPSData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class GPSCollector extends AsynchronousCollector implements LocationListener {
    private LocationManager locationManager;
    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private boolean isProviderEnabled;
    private final GPSData data;
    private final Handler handler;

    /*
      Error code:
        0: No error
        1: GPS provider not enabled
        2: Concurrent task of GPS collecting
        3: Unknown exception when stopping collecting
        4: Invalid GPS request time
     */

    public GPSCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        data = new GPSData();
        handler = new Handler(Looper.getMainLooper());
    }

    @Override
    public void initialize() {
        locationManager = (LocationManager) mContext.getSystemService(Context.LOCATION_SERVICE);
        isProviderEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
    }

    @Override
    public void close() {
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void pause() {
        unbindListener();
    }

    @Override
    public void resume() {
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    private boolean bindListener() {
        if (isProviderEnabled && isRunning.compareAndSet(false, true)) {
            Log.e("TEST", locationManager + " " + mContext + " " );
            locationManager.registerGnssStatusCallback(gnssStatusCallback, handler);
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000, 1, this, Looper.getMainLooper());
            return true;
        } else {
            return false;
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    private boolean unbindListener() {
        if (isProviderEnabled && isRunning.compareAndSet(true, false)) {
            locationManager.unregisterGnssStatusCallback(gnssStatusCallback);
            locationManager.removeUpdates(this);
            return true;
        } else {
            return false;
        }
    }

    @Override
    public String getName() {
        return "GPS";
    }

    @Override
    public String getExt() {
        return ".json";
    }

    @SuppressLint("MissingPermission")
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();

        if (config.getGPSRequestTime() <= 0) {
            result.setErrorCode(4);
            result.setErrorReason("Invalid GPS request time: " + config.getGPSRequestTime());
            ft.complete(result);
        } else if (isProviderEnabled) {
            if (bindListener()) {
                futureList.add(scheduledExecutorService.schedule(() -> {
                    try {
                        setLocation(locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER));
                        unbindListener();
                    } catch (Exception e) {
                        e.printStackTrace();
                        result.setErrorCode(3);
                        result.setErrorReason(e.toString());
                    } finally {
                        setCollectData(result);
                        ft.complete(result);
                        isRunning.set(false);
                    }
                }, config.getGPSRequestTime(), TimeUnit.MILLISECONDS));
            } else {
                result.setErrorCode(2);
                result.setErrorReason("Concurrent task of GPS collecting");
                ft.complete(result);
            }
        } else {
            result.setErrorCode(1);
            result.setErrorReason("GPS provider not enabled");
            ft.complete(result);
        }

        return ft;
    }

    private void setLocation(Location location) {
        if (location != null) {
            data.setAccuracy(location.getAccuracy());
            data.setAltitude(location.getAltitude());
            data.setLatitude(location.getLatitude());
            data.setLongitude(location.getLongitude());
            data.setBearing(location.getBearing());
            data.setSpeed(location.getSpeed());
            data.setTime(location.getTime());
            data.setProvider(location.getProvider());
        }
    }

    @Override
    public void onLocationChanged(@NonNull Location location) {
        synchronized (data) {
            setLocation(location);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private final GnssStatus.Callback gnssStatusCallback = new GnssStatus.Callback() {
        @Override
        public void onStarted() {
            super.onStarted();
        }

        @Override
        public void onStopped() {
            super.onStopped();
        }

        @Override
        public void onFirstFix(int ttffMillis) {
            super.onFirstFix(ttffMillis);
        }

        @RequiresApi(api = Build.VERSION_CODES.R)
        @Override
        public void onSatelliteStatusChanged(@NonNull GnssStatus status) {
            super.onSatelliteStatusChanged(status);
            List<GPSData.SatelliteData> satellites = new ArrayList<>();
            int count = status.getSatelliteCount();
            data.setSatelliteCount(count);
            for(int i = 0; i < count; i++) {
                satellites.add(new GPSData.SatelliteData(
                        status.getConstellationType(i),
                        status.getSvid(i),
                        status.getAzimuthDegrees(i),
                        status.getCarrierFrequencyHz(i),
                        status.getCn0DbHz(i),
                        status.getElevationDegrees(i)));
            }
            synchronized (data) {
                data.setSatellites(satellites);
            }
        }
    };

    private void setCollectData(CollectorResult result) {
        synchronized (data) {
            result.setData(data.deepClone());
            result.setDataString(gson.toJson(result.getData(), GPSData.class));
        }
    }
}
