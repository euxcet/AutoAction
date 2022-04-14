package com.hcifuture.contextactionlibrary.collect.collector;

import android.annotation.SuppressLint;
import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.amap.api.location.AMapLocation;
import com.amap.api.location.AMapLocationClient;
import com.amap.api.location.AMapLocationClientOption;
import com.amap.api.location.AMapLocationListener;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class LocationClient {

    static final String TAG = "LocationClient";

    private volatile static LocationClient mInstance;

    @SuppressLint("StaticFieldLeak")
    private static AMapLocationClient mLocationClient;
    private static AMapLocationListener mListener;

    public volatile AMapLocation mLocation;

    private Context mContext;
    private ThreadPoolExecutor mThreadPoolExecutor;
    private CompletableFuture<AMapLocation> mCurrentFuture;

    private LocationClient(Context context) {
        mContext = context;
        mThreadPoolExecutor = new ThreadPoolExecutor(1, Integer.MAX_VALUE, 30, TimeUnit.SECONDS, new LinkedBlockingDeque<>(2));
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public synchronized CompletableFuture<AMapLocation> requestLocation() {
        return requestLocation(60000);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public synchronized CompletableFuture<AMapLocation> requestLocation(long timeout) {
        if (mCurrentFuture != null) {
            return mCurrentFuture;
        }
        mCurrentFuture = new CompletableFuture<>();
        mThreadPoolExecutor.execute(() -> {
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
                            synchronized (LocationClient.this) {
                                if (mCurrentFuture != null) {
                                    mCurrentFuture.complete(aMapLocation);
                                    stopLocation();
                                }
                                LocationClient.this.notifyAll();
                                return;
                            }
                        }
                    }
                    synchronized (LocationClient.this) {
                        mCurrentFuture.completeExceptionally(aMapLocation != null ? new LocationException(aMapLocation.getErrorCode(), aMapLocation.getErrorInfo()) : new NullPointerException("receive null aMapLocation"));
                        stopLocation();
                        LocationClient.this.notifyAll();
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
            synchronized (LocationClient.this) {
                try {
                    LocationClient.this.wait(timeout);
                } catch (Exception e) {
                    e.printStackTrace();
                    if (mCurrentFuture != null) {
                        mCurrentFuture.completeExceptionally(e);
                        stopLocation();
                    }
                }
            }
        });
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

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<AMapLocation> getLocation() {
       CompletableFuture<AMapLocation> ft = new CompletableFuture<>();
       if (mLocation != null) {
           ft.complete(mLocation);
       } else {
           ft = requestLocation();
       }
       return ft;
    }

    public static LocationClient getInstance(Context mContext) {
        if (mInstance == null) {
            synchronized (LocationClient.class) {
                if (mInstance == null) {
                    mInstance = new LocationClient(mContext);
                }
            }
        }
        return mInstance;
    }
}
