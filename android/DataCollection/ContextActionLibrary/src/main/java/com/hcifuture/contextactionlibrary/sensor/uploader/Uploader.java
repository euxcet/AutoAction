package com.hcifuture.contextactionlibrary.sensor.uploader;

import android.annotation.SuppressLint;
import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Build;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.hcifuture.contextactionlibrary.utils.FileUtils;
import com.hcifuture.contextactionlibrary.utils.NetworkUtils;

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

    private static final Gson gson = new Gson();

    private Context mContext;

    private final Lock lock = new ReentrantLock();
    private final Condition compressCondition = lock.newCondition();
    private final Condition uploadCondition = lock.newCondition();

    private final PriorityQueue<UploadTask> compressQueue = new PriorityQueue<>();
    private final PriorityQueue<UploadTask> uploadQueue = new PriorityQueue<>();
    private final int QUEUE_ELEMENT_LIMIT = 10000;

    private ScheduledExecutorService scheduledExecutorService;
    private List<ScheduledFuture<?>> futureList;

    private ScheduledFuture<?> uploadFuture;
    private ScheduledFuture<?> compressFuture;
    private ScheduledFuture<?> localFuture;

    private AtomicBoolean isRunning = new AtomicBoolean(false);

    private String fileFolder;
    private String zipFolder;

    enum UploaderStatus {
        OK,
        QUEUE_IS_FULL,
        CAN_NOT_GET_LOCK
    }

    public Uploader(Context context,
                    ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.mContext = context;
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
        this.fileFolder = mContext.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Click/";
        this.zipFolder = mContext.getExternalMediaDirs()[0].getAbsolutePath() + "/Data/Zip/";
        start();
    }

    private void start() {
        stop();
        isRunning.set(true);
        uploadFuture = scheduledExecutorService.schedule(this::upload, 0, TimeUnit.MILLISECONDS);
        compressFuture = scheduledExecutorService.schedule(this::compress, 0, TimeUnit.MILLISECONDS);
        localFuture = scheduledExecutorService.schedule(this::uploadLocalFiles, 0, TimeUnit.MILLISECONDS);
        futureList.add(uploadFuture);
        futureList.add(compressFuture);
        futureList.add(localFuture);
    }

    private void stop() {
        if (uploadFuture != null) {
            uploadFuture.cancel(true);
        }
        if (compressFuture != null) {
            compressFuture.cancel(true);
        }
        if (localFuture != null) {
            localFuture.cancel(true);
        }
        isRunning.set(false);
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

    public UploaderStatus pushTask(UploadTask task, boolean needCompression) {
        if (needCompression) {
            try {
                lock.lock();
                if (compressQueue.size() >= QUEUE_ELEMENT_LIMIT) {
                    lock.unlock();
                    return UploaderStatus.QUEUE_IS_FULL;
                }
                compressQueue.add(task);
                compressCondition.signal();
            } finally {
                lock.unlock();
            }
        } else {
            try {
                lock.lock();
                if (uploadQueue.size() >= QUEUE_ELEMENT_LIMIT) {
                    lock.unlock();
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
        while (isRunning.get()) {
            List<UploadTask> pack = new ArrayList<>();
            try {
                lock.lock();
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
                String zipName = System.currentTimeMillis() + ".zip";
                String metaName = zipName + ".meta";
                try {
                    File zipFile = new File(zipFolder + zipName);
                    File metaFile = new File(zipFolder + metaName);
                    FileUtils.makeFile(zipFile);
                    FileUtils.makeFile(metaFile);
                    ZipOutputStream os = new ZipOutputStream(new FileOutputStream(zipFile));
                    for (int i = 0; i < pack.size(); i++) {
                        File file = pack.get(i).getFile();
                        Log.d(TAG, "[PACK] Compressing " + file.getAbsolutePath());
                        ZipEntry zipEntry = new ZipEntry(file.getName());
                        FileInputStream is = new FileInputStream(file);
                        os.putNextEntry(zipEntry);
                        int len;
                        byte[] buffer = new byte[1024 * 1024];
                        while ((len = is.read(buffer)) != -1) {
                            os.write(buffer, 0, len);
                        }
                        is.close();
                        os.closeEntry();
                    }
                    os.finish();
                    os.close();
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                        List<TaskMetaBean> metas = pack.stream().map((x) -> x.getMeta().get(0)).collect(Collectors.toList());
                        FileUtils.writeStringToFile(gson.toJson(metas), metaFile);
                        pushTask(new UploadTask(zipFile, metaFile, metas), false);
                    }
                    for (int i = 0; i < pack.size(); i++) {
                        FileUtils.deleteFile(pack.get(i).getFile(), "PACK");
                        FileUtils.deleteFile(pack.get(i).getMetaFile(), "PACK");
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void upload() {
        while (isRunning.get()) {
            UploadTask task;
            try {
                lock.lock();
                if (uploadQueue.isEmpty()) {
                    try {
                        uploadCondition.await();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                if (!isUnderWifi()) {
                    lock.unlock();
                    continue;
                }
                task = uploadQueue.poll();
            } finally {
                lock.unlock();
            }

            if (task != null) {
                NetworkUtils.uploadCollectedData(task, new Callback() {
                    @Override
                    public void onFailure(@NonNull Call call, @NonNull IOException e) {
                        if (task.getRemainingRetries() > 0) {
                            task.setRemainingRetries(task.getRemainingRetries() - 1);
                            task.setExpectedUploadTime(task.getExpectedUploadTime() + HOUR);
                            if (pushTask(task, true) == UploaderStatus.QUEUE_IS_FULL) {
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
                        FileUtils.deleteFile(task.getFile(), "UPLOAD");
                        FileUtils.deleteFile(task.getMetaFile(), "UPLOAD");
                    }
                });
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
                                            pushTask(new UploadTask(file, metaFile, meta), false);
                                        }
                                    }
                                } else {
                                    String content = FileUtils.getFileContent(metaFile.getAbsolutePath());
                                    TaskMetaBean meta = gson.fromJson(content, TaskMetaBean.class);
                                    if (meta.getTimestamp() < timestamp) {
                                        pushTask(new UploadTask(file, metaFile, meta), true);
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
        long timestamp = System.currentTimeMillis();
        uploadDirectory(new File(this.fileFolder), timestamp, false);
        uploadDirectory(new File(this.zipFolder), timestamp, true);
    }
}
