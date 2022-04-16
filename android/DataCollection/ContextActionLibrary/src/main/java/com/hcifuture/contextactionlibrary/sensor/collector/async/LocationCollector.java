package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.data.Data;
import com.hcifuture.contextactionlibrary.sensor.data.LocationData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class LocationCollector extends AsynchronousCollector {
    private LocationClient client;

    private LocationData data;

    public LocationCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<Data> getData(TriggerConfig config) {
        if (client != null) {
            return client.requestLocation().thenApply(aMapLocation -> {
                data = new LocationData(
                        aMapLocation.getLongitude(),
                        aMapLocation.getLatitude(),
                        aMapLocation.getAltitude(),
                        aMapLocation.getAccuracy(),
                        aMapLocation.getFloor(),
                        aMapLocation.getCity(),
                        aMapLocation.getPoiName(),
                        aMapLocation.getStreet(),
                        aMapLocation.getTime(),
                        aMapLocation.getAdCode(),
                        aMapLocation.getCityCode()
                );
                return data;
            });
        } else {
            return CompletableFuture.completedFuture(null);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<String> getDataString(TriggerConfig config) {
        return getData(config).thenApply((d) -> d == null ? null : gson.toJson(d, LocationData.class));
    }

    @Override
    public void initialize() {
        client = LocationClient.getInstance(mContext);
    }

    /*
    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Void> collect(TriggerConfig config) {
        if (client != null) {
            return client.requestLocation().thenApply(aMapLocation -> {
                data = new LocationData(
                        aMapLocation.getLongitude(),
                        aMapLocation.getLatitude(),
                        aMapLocation.getAltitude(),
                        aMapLocation.getAccuracy(),
                        aMapLocation.getFloor(),
                        aMapLocation.getCity(),
                        aMapLocation.getPoiName(),
                        aMapLocation.getStreet(),
                        aMapLocation.getTime(),
                        aMapLocation.getAdCode(),
                        aMapLocation.getCityCode()
                );
                return data;
            }).thenCompose(data -> saver.save(data));
        } else {
            return CompletableFuture.completedFuture(null);
        }
    }
     */

    @Override
    public void close() {
    }

    @Override
    public synchronized void pause() {
    }

    @Override
    public synchronized void resume() {
    }

    @Override
    public String getName() {
        return "Location";
    }

    @Override
    public String getExt() {
        return ".txt";
    }
}
