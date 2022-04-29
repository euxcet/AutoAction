package com.hcifuture.contextactionlibrary.sensor.uploader;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.util.Log;

import androidx.annotation.NonNull;

import com.hcifuture.contextactionlibrary.utils.NetworkUtils;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.Queue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;

public class Uploader {
    private static final String TAG = "Uploader";

    private Context mContext;

    private final Lock lock = new ReentrantLock();
    private final Condition condition = lock.newCondition();

    private final Queue<UploadTask> pendingQueue = new LinkedList<>();
    private final int QUEUE_ELEMENT_LIMIT = 10000;

    private ScheduledExecutorService scheduledExecutorService;
    private List<ScheduledFuture<?>> futureList;

    private ScheduledFuture<?> uploadFuture;

    private AtomicBoolean isRunning = new AtomicBoolean(false);

    enum UploaderStatus {
        OK,
        CAN_NOT_GET_LOCK,
        PENDING_QUEUE_IS_FULL
    }

    public Uploader(Context context,
                    ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        start();
    }

    private void start() {
        stop();
        uploadFuture = scheduledExecutorService.schedule(this::process, 0, TimeUnit.MILLISECONDS);
        futureList.add(uploadFuture);
        isRunning.set(true);
    }

    private void stop() {
        if (uploadFuture != null) {
            uploadFuture.cancel(true);
        }
        isRunning.set(false);
    }

    public UploaderStatus pushTask(UploadTask task) {
        Log.e(TAG, "New Task");
        try {
            lock.lock();
            if (pendingQueue.size() >= QUEUE_ELEMENT_LIMIT) {
                lock.unlock();
                return UploaderStatus.PENDING_QUEUE_IS_FULL;
            }
            pendingQueue.add(task);
            condition.signal();
        } finally {
            lock.unlock();
        }
        return UploaderStatus.OK;
    }

    private void process() {
        while (isRunning.get()) {
            try {
                lock.lock();
                if (pendingQueue.isEmpty()) {
                    try {
                        condition.await();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                Log.e(TAG, "Get condition " + isUnderWifi());

                if (!isUnderWifi()) {
                    lock.unlock();
                    continue;
                }
            } finally {
                lock.unlock();
            }

            UploadTask task = pendingQueue.poll();

            if (task != null) {
                NetworkUtils.uploadCollectedData(task, new Callback() {
                    @Override
                    public void onFailure(@NonNull Call call, @NonNull IOException e) {
                        if (task.getRemainingRetries() > 0) {
                            task.setRemainingRetries(task.getRemainingRetries() - 1);
                            if (pushTask(task) == UploaderStatus.PENDING_QUEUE_IS_FULL) {
                                Log.e(TAG, task.getFile().getAbsolutePath()
                                        + " could not be uploaded because the queue is full");
                            }
                        } else {
                            Log.e(TAG, task.getFile().getAbsolutePath()
                                    + " could not be uploaded because the maximum number of retries is reached");
                        }
                    }

                    @Override
                    public void onResponse(@NonNull Call call, @NonNull Response response) {
                        Log.d(TAG, "Successfully uploaded " + task.getFile().getAbsolutePath());
                    }
                });
            }
        }
    }

    private boolean isUnderWifi() {
        ConnectivityManager connectivityManager = (ConnectivityManager) mContext
                .getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo info = connectivityManager.getActiveNetworkInfo();
        return info != null && info.getType() == ConnectivityManager.TYPE_WIFI;
    }
}
