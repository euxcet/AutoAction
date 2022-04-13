package com.hcifuture.contextactionlibrary.collect.collector;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.hcifuture.contextactionlibrary.collect.data.Data;
import com.hcifuture.contextactionlibrary.collect.data.LocationData;
import com.hcifuture.contextactionlibrary.collect.trigger.Trigger;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;

public class LocationCollector extends Collector {
    private LocationClient client;

    private LocationData data;

    public LocationCollector(Context context, Trigger.CollectorType type, String triggerFolder, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, triggerFolder, scheduledExecutorService, futureList);
        data = new LocationData();
    }

    @Override
    public void initialize() {
        client = LocationClient.getInstance(mContext);
    }

    @Override
    public void setSavePath(String timestamp) {
        if (data instanceof java.util.List) {
            saver.setSavePath(timestamp + "_location.bin");
        }
        else {
            saver.setSavePath(timestamp + "_location.txt");
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public synchronized CompletableFuture<Void> collect() {
        Log.e("Location", "" + (client.mLocation == null));
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
                Log.e("Location", "result " + data.getLatitude() + " " + data.getLongitude());
                return data;
            }).thenCompose(data -> saver.save(data));
        } else {
            return CompletableFuture.completedFuture(null);
        }
    }

    @Override
    public void close() {
    }

    @Override
    public boolean forPrediction() {
        return true;
    }

    @Override
    public synchronized Data getData() {
        Gson gson = new Gson();
        return gson.fromJson(gson.toJson(data), LocationData.class);
    }

    @Override
    public String getSaveFolderName() {
        return "Location";
    }

    @Override
    public synchronized void pause() {
    }

    @Override
    public synchronized void resume() {
    }
}
