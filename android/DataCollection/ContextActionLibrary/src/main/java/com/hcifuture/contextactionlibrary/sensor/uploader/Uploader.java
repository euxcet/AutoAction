package com.hcifuture.contextactionlibrary.sensor.uploader;

import android.annotation.SuppressLint;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Handler;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.ToNumberPolicy;
import com.google.gson.ToNumberStrategy;
import com.google.gson.reflect.TypeToken;
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;
import com.hcifuture.shared.communicate.config.RequestConfig;
import com.hcifuture.shared.communicate.listener.RequestListener;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.Response;

public class Uploader {
    private static final String TAG = "Uploader";
    private static final long SECOND = 1000;
    private static final long MINUTE = SECOND * 60;
    private static final long HOUR = MINUTE * 60;
    private static final int FILES_IN_PACKAGE = 5;

    private static final Gson gson = new GsonBuilder()
            .disableHtmlEscaping()
            .setObjectToNumberStrategy(ToNumberPolicy.LONG_OR_DOUBLE)
            .create();

    private final Context mContext;

    private final Lock lock = new ReentrantLock();
    private final Condition compressCondition = lock.newCondition();
    private final Condition uploadCondition = lock.newCondition();

    private final PriorityQueue<UploadTask> compressQueue = new PriorityQueue<>();
    private final PriorityQueue<UploadTask> uploadQueue = new PriorityQueue<>();
    private final int QUEUE_ELEMENT_LIMIT = 10000;

    private final ScheduledExecutorService scheduledExecutorService;
    private final List<ScheduledFuture<?>> futureList;

    private final List<ScheduledFuture<?>> mFutureList = new ArrayList<>();

    private final AtomicBoolean isRunning = new AtomicBoolean(false);
    private final AtomicBoolean isUploadingLocalFiles = new AtomicBoolean(false);

    private final String fileFolder;
    private final String zipFolder;
    private final AtomicInteger mZipIDCounter = new AtomicInteger(0);
    private final RequestListener requestListener;

    private boolean lastWifiStatus = false;
    private final BroadcastReceiver receiver;
    private final Handler handler;

    enum UploaderStatus {
        OK,
        QUEUE_IS_FULL,
        CAN_NOT_GET_LOCK
    }

    public Uploader(Context context,
                    ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList,
                    RequestListener requestListener, Handler handler) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        this.fileFolder = mContext.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Click/";
        this.zipFolder = mContext.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Zip/";
        this.requestListener = requestListener;
        this.handler = handler;
        this.receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                NetworkInfo info = intent.getParcelableExtra(WifiManager.EXTRA_NETWORK_INFO);
                if (info != null) {
                    if (info.isConnected()) {
                        if (!lastWifiStatus) {
                            lastWifiStatus = true;
                            Log.e(TAG, "broadcast receive: Wifi available now");
                            addFuture(scheduledExecutorService.schedule(Uploader.this::uploadLocalFiles, 0, TimeUnit.MILLISECONDS));
                        }
                    } else {
                        lastWifiStatus = false;
                    }
                }
            }
        };
        start();
    }

    private void start() {
        stop();
        mContext.registerReceiver(receiver, new IntentFilter(WifiManager.NETWORK_STATE_CHANGED_ACTION), null, handler);
        isRunning.set(true);
        addFuture(scheduledExecutorService.schedule(this::upload, 0, TimeUnit.MILLISECONDS));
        addFuture(scheduledExecutorService.schedule(this::compress, 0, TimeUnit.MILLISECONDS));
        addFuture(scheduledExecutorService.scheduleWithFixedDelay(this::uploadLocalFiles, 0, 12 * HOUR, TimeUnit.MILLISECONDS));
    }

    private synchronized void addFuture(ScheduledFuture<?> future) {
        mFutureList.add(future);
        futureList.add(future);
    }

    public void stop() {
        isRunning.set(false);
        for (ScheduledFuture<?> future: mFutureList) {
            future.cancel(true);
        }
        try {
            if (receiver != null) {
                mContext.unregisterReceiver(receiver);
            }
        } catch (Exception ignored) {
        }
    }

    public UploaderStatus flush() {
        try {
            lock.lock();
            while (!compressQueue.isEmpty()) {
                uploadQueue.add(compressQueue.poll());
            }
            uploadCondition.signal();
        } finally {
            lock.unlock();
        }
        return UploaderStatus.OK;
    }

    public UploaderStatus pushTask(UploadTask task) {
        if (task.isNeedCompression()) {
            lock.lock();
            try {
                if (compressQueue.size() >= QUEUE_ELEMENT_LIMIT) {
                    return UploaderStatus.QUEUE_IS_FULL;
                }
                compressQueue.add(task);
                compressCondition.signal();
            } finally {
                lock.unlock();
            }
        } else {
            lock.lock();
            try {
                if (uploadQueue.size() >= QUEUE_ELEMENT_LIMIT) {
                    return UploaderStatus.QUEUE_IS_FULL;
                }
                uploadQueue.add(task);
                uploadCondition.signal();
            } finally {
                lock.unlock();
            }
        }
        return UploaderStatus.OK;
    }

    private void compress() {
        List<File> needToDelete = new ArrayList<>(FILES_IN_PACKAGE * 2);
        List<UploadTask> pack = new ArrayList<>(FILES_IN_PACKAGE);
        while (!Thread.currentThread().isInterrupted() && isRunning.get()) {
            try {
                pack.clear();
                needToDelete.clear();
                lock.lock();
                try {
                    if (compressQueue.size() < FILES_IN_PACKAGE) {
                        try {
                            compressCondition.await();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    if (compressQueue.size() >= FILES_IN_PACKAGE) {
                        for (int i = 0; i < FILES_IN_PACKAGE; i++) {
                            pack.add(compressQueue.poll());
                        }
                    }
                } finally {
                    lock.unlock();
                }

                if (pack.size() > 0) {
                    try {
                        /*
                          for each zip file:
                            1. must assign user ID to avoid naming conflicts on server
                            2. must assign unique ID to avoid naming conflicts on local device
                         */
                        String zipName = getUserId() + "_" + System.currentTimeMillis() + "_" + mZipIDCounter.getAndIncrement() + ".zip";
                        String metaName = zipName + ".meta";
                        File zipFile = new File(zipFolder + zipName);
                        File metaFile = new File(zipFolder + metaName);
                        FileUtils.makeFile(zipFile);
                        FileUtils.makeFile(metaFile);
                        List<TaskMetaBean> metas = new ArrayList<>(FILES_IN_PACKAGE);
                        try (ZipOutputStream os = new ZipOutputStream(new FileOutputStream(zipFile))) {
                            for (int i = 0; i < pack.size(); i++) {
                                File file = pack.get(i).getFile();
                                Log.e(TAG, "[PACK] Compressing " + file.getAbsolutePath());
                                ZipEntry zipEntry = new ZipEntry(file.getName());
                                // may throw FileNotFoundException
                                try (FileInputStream is = new FileInputStream(file)) {
                                    os.putNextEntry(zipEntry);
                                    int len;
                                    byte[] buffer = new byte[1024 * 1024];
                                    while ((len = is.read(buffer)) != -1) {
                                        os.write(buffer, 0, len);
                                    }
                                    os.closeEntry();
                                    // add meta info and mark already packed files for later deletion
                                    metas.add(pack.get(i).getMeta().get(0));
                                    needToDelete.add(file);
                                    needToDelete.add(pack.get(i).getMetaFile());
                                } catch (Exception e) {
                                    e.printStackTrace();
                                }
                            }
                        }
                        Log.e(TAG, "compress packed entries: " + metas.size());
                        Log.e(TAG, "compress zip filename: " + zipName);
                        if (metas.size() > 0) {
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
//                                List<TaskMetaBean> metas = pack.stream().map((x) -> x.getMeta().get(0)).collect(Collectors.toList());
                                FileUtils.writeStringToFile(gson.toJson(metas), metaFile);
                                pushTask(new UploadTask(zipFile, metaFile, metas, false));
                            }
                            for (int i = 0; i < needToDelete.size(); i++) {
                                FileUtils.deleteFile(needToDelete.get(i), "PACK");
                            }
                        } else {
                            // pack contains no file, delete .zip and .zip.meta
                            FileUtils.deleteFile(zipFile, "PACK FAIL");
                            FileUtils.deleteFile(metaFile, "PACK FAIL");
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    private void upload() {
        while (!Thread.currentThread().isInterrupted() && isRunning.get()) {
            try {
                UploadTask task;
                lock.lock();
                try {
                    if (uploadQueue.isEmpty()) {
                        try {
                            uploadCondition.await();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                    if (!isUnderWifi()) {
                        continue;
                    }
                    task = uploadQueue.poll();
                } catch (Exception e) {
                    e.printStackTrace();
                    continue;
                } finally {
                    lock.unlock();
                }

//                if (!lastWifiStatus) {
//                    addFuture(scheduledExecutorService.schedule(this::uploadLocalFiles, 0, TimeUnit.MILLISECONDS));
//                    lastWifiStatus = true;
//                    Log.e(TAG, "upload: Wifi available now");
//                }

                if (task != null) {
                    try {
                        if (task.getFile().exists()) {
                            NetworkUtils.uploadCollectedData(task, new Callback() {
                                @Override
                                public void onFailure(@NonNull Call call, @NonNull IOException e) {
                                    if (task.getRemainingRetries() > 0) {
                                        task.setRemainingRetries(task.getRemainingRetries() - 1);
                                        task.setExpectedUploadTime(task.getExpectedUploadTime() + HOUR);
                                        if (pushTask(task) == UploaderStatus.QUEUE_IS_FULL) {
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
                                    if (response.isSuccessful()) {
                                        Log.d(TAG, "Successfully uploaded " + task.getFile().getAbsolutePath());
                                        FileUtils.deleteFile(task.getFile(), "UPLOAD");
                                        FileUtils.deleteFile(task.getMetaFile(), "UPLOAD");
                                    }
                                }
                            });
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    @SuppressLint("MissingPermission")
    private boolean isUnderWifi() {
        ConnectivityManager connectivityManager = (ConnectivityManager) mContext
                .getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo info = connectivityManager.getActiveNetworkInfo();
        return info != null && info.getType() == ConnectivityManager.TYPE_WIFI;
    }

    private void uploadDirectory(File dir, long timestamp, boolean isZip) {
        Log.e(TAG, "uploadDirectory: " + dir + " isZip: " + isZip);
        if (dir.exists()) {
            File[] files = dir.listFiles();
            if (files != null) {
                for (File file : files) {
                    try {
                        Log.e(TAG, "uploadDirectory: checking file: " + file);
                        if (file.isDirectory()) {
                            uploadDirectory(file, timestamp, isZip);
                        } else {
                            File metaFile = new File(file.getAbsolutePath() + ".meta");
                            if (metaFile.exists()) {
                                if (isZip) {
                                    Type type = new TypeToken<List<TaskMetaBean>>(){}.getType();
                                    String content = FileUtils.getFileContent(metaFile.getAbsolutePath());
                                    List<TaskMetaBean> meta = gson.fromJson(content, type);
                                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                                        if (meta.stream().noneMatch(v -> v.getTimestamp() >= timestamp)) {
                                            pushTask(new UploadTask(file, metaFile, meta, false));
                                        }
                                    }
                                } else {
                                    String content = FileUtils.getFileContent(metaFile.getAbsolutePath());
                                    TaskMetaBean meta = gson.fromJson(content, TaskMetaBean.class);
                                    if (meta.getTimestamp() < timestamp) {
                                        pushTask(new UploadTask(file, metaFile, meta, true));
                                    }
                                }
                            }
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    private void uploadLocalFiles() {
        Log.e(TAG, "try to uploadLocalFiles");
        if (isRunning.get()) {
            if (isUploadingLocalFiles.compareAndSet(false, true)) {
                try {
                    Log.e(TAG, "uploadLocalFiles");
                    long timestamp = System.currentTimeMillis() - 5 * MINUTE;
                    uploadDirectory(new File(this.fileFolder), timestamp, false);
                    uploadDirectory(new File(this.zipFolder), timestamp, true);
                } finally {
                    isUploadingLocalFiles.set(false);
                }
            }
        }
    }

    public String getUserId() {
        // get unique user ID
        RequestConfig request = new RequestConfig();
        request.putString("getDeviceId", "");
        String userId = (String) requestListener.onRequest(request).getObject("getDeviceId");
        if (userId == null || "Unknown".equals(userId)) {
            userId = "Unknown_" + System.currentTimeMillis();
        }
        return userId;
    }
}
