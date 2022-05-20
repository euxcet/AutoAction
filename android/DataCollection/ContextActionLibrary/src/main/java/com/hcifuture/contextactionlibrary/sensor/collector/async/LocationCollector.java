package com.hcifuture.contextactionlibrary.sensor.collector.async;

import android.content.Context;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.amap.api.location.AMapLocation;
import com.amap.api.location.AMapLocationClient;
import com.amap.api.location.AMapLocationClientOption;
import com.amap.api.location.AMapLocationListener;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorException;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorManager;
import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.LocationData;
import com.hcifuture.contextactionlibrary.sensor.trigger.TriggerConfig;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class LocationCollector extends AsynchronousCollector {

    private LocationData data;
    private CompletableFuture<AMapLocation> mCurrentFuture;
    private AMapLocationClient mLocationClient;
    private AMapLocationListener mListener;
    public volatile AMapLocation mLocation;

    private final Object waitLock;

    /*
      Error code:
        0: No error
        1: Null LocationClient
        2: Error in client.requestLocation()
     */

    public LocationCollector(Context context, CollectorManager.CollectorType type, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        super(context, type, scheduledExecutorService, futureList);
        waitLock = new Object();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public CompletableFuture<CollectorResult> getData(TriggerConfig config) {
        if (config.getLocationTimeout() <= 0) {
            CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
            ft.completeExceptionally(new Exception("Invalid location timeout: " + config.getLocationTimeout()));
            return ft;
        }

        CollectorResult result = new CollectorResult();
        return requestLocation(config.getLocationTimeout()).thenApply((aMapLocation) -> {
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
            result.setData(data.deepClone());
            result.setDataString(gson.toJson(result.getData(), LocationData.class));
            return result;
        });
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public synchronized CompletableFuture<AMapLocation> requestLocation(long timeout) {
        if (mCurrentFuture != null) {
            return mCurrentFuture;
        }
        mCurrentFuture = new CompletableFuture<>();
        futureList.add(scheduledExecutorService.schedule(() -> {
            AMapLocationClientOption mLocationOption = new AMapLocationClientOption();
            mLocationOption.setLocationMode(AMapLocationClientOption.AMapLocationMode.Hight_Accuracy);
            mLocationOption.setNeedAddress(true);
            mLocationOption.setOnceLocation(true);
            mLocationOption.setInterval(600000);
            try {
                mLocationClient = new AMapLocationClient(mContext);
                mLocationClient.disableBackgroundLocation(true);
                mLocationClient.setLocationOption(mLocationOption);
                mListener = aMapLocation -> {
                    if (aMapLocation != null) {
                        if (Math.abs(aMapLocation.getLatitude()) > 0.1f) {
                            mLocation = aMapLocation.clone();
                            synchronized (waitLock) {
                                if (mCurrentFuture != null) {
                                    mCurrentFuture.complete(aMapLocation);
                                    stopLocation();
                                }
                                waitLock.notifyAll();
                                return;
                            }
                        }
                    }
                    synchronized (waitLock) {
                        mCurrentFuture.completeExceptionally(aMapLocation != null ? new CollectorException(aMapLocation.getErrorCode(), aMapLocation.getErrorInfo()) : new NullPointerException("receive null aMapLocation"));
                        stopLocation();
                        waitLock.notifyAll();
                    }
                };
                mLocationClient.setLocationListener(mListener);
                mLocationClient.startLocation();
            } catch (Exception e) {
                e.printStackTrace();
                if (mCurrentFuture != null) {
                    mCurrentFuture.completeExceptionally(e);
                }
            }
            synchronized (waitLock) {
                try {
                    waitLock.wait(timeout);
                } catch (Exception e) {
                    e.printStackTrace();
                    if (mCurrentFuture != null) {
                        mCurrentFuture.completeExceptionally(e);
                        stopLocation();
                    }
                }
            }
        }, 0, TimeUnit.MILLISECONDS));
        return mCurrentFuture;
    }

    private synchronized void stopLocation() {
        if (mLocationClient != null) {
            mLocationClient.unRegisterLocationListener(mListener);
            mLocationClient.stopLocation();
            mLocationClient.onDestroy();
            mLocationClient = null;
        }
        if (mCurrentFuture != null) {
            mCurrentFuture = null;
        }
    }

    @Override
    public void initialize() {
    }

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
        return ".json";
    }
}
