package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.annotation.SuppressLint;
import android.content.Context;
import android.location.GnssStatus;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorException;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.GPSData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;
import com.hcifuture.contextactionlibrary.status.Heart;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public class GPSCollector extends AsynchronousCollector implements LocationListener {
    private LocationManager locationManager;
    private final AtomicBoolean isCollecting = new AtomicBoolean(false);
    private boolean isProviderEnabled;
    private final GPSData data;

    /*
      Error code:
        0: No error
        1: GPS provider not enabled
        2: Concurrent task of GPS collecting
        3: Unknown exception when stopping collecting
        4: Invalid GPS request time
        5: Unknown collecting exception (may be binding listener error due to lack of permission)
     */

    public GPSCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        data = new GPSData();
    }

    @Override
    public void initialize() {
        locationManager = (LocationManager) mContext.getSystemService(Context.LOCATION_SERVICE);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public void close() {
        unbindListener();
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
    private void bindListener() {
        Log.e("TEST", locationManager + " " + mContext + " " );
        locationManager.registerGnssStatusCallback(gnssStatusCallback, handler);
        locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000, 1, this, handler.getLooper());
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @SuppressLint("MissingPermission")
    private void unbindListener() {
        try {
            locationManager.removeUpdates(this);
            locationManager.unregisterGnssStatusCallback(gnssStatusCallback);
        } catch (Exception e) {
            e.printStackTrace();
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
        Heart.getInstance().newSensorGetEvent(getName(), System.currentTimeMillis());
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        CollectorResult result = new CollectorResult();

        if (config.getGPSRequestTime() <= 0) {
            ft.completeExceptionally(new CollectorException(4, "Invalid GPS request time: " + config.getGPSRequestTime()));
        } else if (isCollecting.compareAndSet(false, true)) {
            try {
                setBasicInfo();
                if (!isProviderEnabled) {
                    setCollectData(result);
                    result.setErrorCode(1);
                    result.setErrorReason("GPS provider not enabled");
                    isCollecting.set(false);
                    ft.complete(result);
                } else {
                    bindListener();
                    futureList.add(scheduledExecutorService.schedule(() -> {
                        try {
                            setLocation(locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER));
                        } catch (Exception e) {
                            e.printStackTrace();
                            result.setErrorCode(3);
                            result.setErrorReason(e.toString());
                        } finally {
                            unbindListener();
                            setCollectData(result);
                            isCollecting.set(false);
                            ft.complete(result);
                        }
                    }, config.getGPSRequestTime(), TimeUnit.MILLISECONDS));
                }
            } catch (Exception e) {
                e.printStackTrace();
                unbindListener();
                isCollecting.set(false);
                ft.completeExceptionally(new CollectorException(5, e));
            }
        } else {
            ft.completeExceptionally(new CollectorException(2, "Concurrent task of GPS collecting"));
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

    @Override
    public void onProviderEnabled(@NonNull java.lang.String provider) {

    }

    @Override
    public void onProviderDisabled(@NonNull java.lang.String provider) {

    }

    @Override
    public void onStatusChanged(String provider, int status, android.os.Bundle extras) {

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

    private void setBasicInfo() {
        if (locationManager != null) {
            isProviderEnabled = locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER);
            data.setProviderEnabled(isProviderEnabled);
        }
    }
}
