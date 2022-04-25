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
    private Handler handler;

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

    @Override
    public void pause() {
        unbindListener();
    }

    @Override
    public void resume() {
    }

    private void bindListener() {
        if (isProviderEnabled && !isRunning.get()) {
            Log.e("TEST", locationManager + " " + mContext + " " );
            locationManager.registerGnssStatusCallback(gnssStatusCallback, handler);
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1000, 1, this, Looper.getMainLooper());
            isRunning.set(true);
        }
    }

    private void unbindListener() {
        if (isProviderEnabled && isRunning.get()) {
            locationManager.unregisterGnssStatusCallback(gnssStatusCallback);
            locationManager.removeUpdates(this);
            isRunning.set(false);
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        if (isProviderEnabled) {
            if (!isRunning.get()) {
                bindListener();
                scheduledExecutorService.schedule(() -> {
                    try {
                        setLocation(locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER));
                        unbindListener();
                        CollectorResult result = new CollectorResult();
                        synchronized (data) {
                            result.setData(data.deepClone());
                            result.setDataString(gson.toJson(result.getData(), GPSData.class));
                            ft.complete(result);
                        }
                    } catch (Exception e) {
                        ft.completeExceptionally(e);
                    } finally {
                        isRunning.set(false);
                    }
                }, config.getGPSRequestTime(), TimeUnit.MILLISECONDS);
            } else {
                ft.completeExceptionally(new Exception("Another task of GPS recording is taking place!"));
            }
        } else {
            ft.completeExceptionally(new Exception("GPS provider is not enabled"));
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
}
