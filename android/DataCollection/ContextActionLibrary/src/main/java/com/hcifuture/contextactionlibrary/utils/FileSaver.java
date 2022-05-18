package com.hcifuture.contextactionlibrary.utils;

import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.IMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class FileSaver {
    private static volatile FileSaver instance = null;
    private final Lock lock = new ReentrantLock();
    private final Condition condition = lock.newCondition();
    private final Queue<SaveTask> queue = new LinkedList<>();
    private final AtomicBoolean isRunning;

    private FileSaver(ScheduledExecutorService service, List<ScheduledFuture<?>> futureList) {
        isRunning = new AtomicBoolean(true);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            futureList.add(service.schedule(this::save, 0, TimeUnit.MILLISECONDS));
        }
    }

    public void close() {
        isRunning.set(false);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void saveString(SaveTask task) {
        try {
            File saveFile = task.getSaveFile();
            CollectorResult result = task.getResult();
            CompletableFuture<CollectorResult> ft = task.getFuture();
            Log.e("TEST", "saveString " + saveFile.getAbsolutePath());

            FileUtils.makeFile(saveFile);
            String toWrite = result.getDataString() + "\r\n";
            OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(saveFile));
            writer.write(toWrite);
            writer.close();
            result.setSavePath(saveFile.getAbsolutePath());
            ft.complete(result);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void saveIMUData(SaveTask task) {
        try {
            File saveFile = task.getSaveFile();
            CollectorResult result = task.getResult();
            CompletableFuture<CollectorResult> ft = task.getFuture();
            assert result.getData() instanceof IMUData;

            FileUtils.makeFile(saveFile);
            FileOutputStream fos = new FileOutputStream(saveFile);
            DataOutputStream dos = new DataOutputStream(fos);
            for (SingleIMUData d: ((IMUData)result.getData()).getData()) {
                List<Float> values = d.getValues();
                dos.writeFloat(values.get(0));
                dos.writeFloat(values.get(1));
                dos.writeFloat(values.get(2));
                dos.writeFloat((float)d.getType());
                dos.writeDouble((double)d.getTimestamp());
            }
            dos.flush();
            dos.close();
            fos.flush();
            fos.close();

            result.setSavePath(saveFile.getAbsolutePath());

            ft.complete(result);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void save() {
        while (!Thread.currentThread().isInterrupted() && isRunning.get()) {
            lock.lock();
            SaveTask task;
            try {
                if (queue.isEmpty()) {
                    try {
                        condition.await();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                task = queue.poll();
            } finally {
                lock.unlock();
            }
            if (task != null) {
                if (task.getType() == 0) {
                    saveString(task);
                } else {
                    saveIMUData(task);
                }
            }
        }
    }

    private void pushTask(SaveTask task) {
        lock.lock();
        try {
            queue.add(task);
            condition.signal();
        } finally {
            lock.unlock();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> writeStringToFile(CollectorResult result, File saveFile) {
        SaveTask task = new SaveTask(saveFile, result, new CompletableFuture<>(), 0);
        Log.e("TEST", "new Task " + saveFile.getAbsolutePath());
        pushTask(task);
        return task.getFuture();
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<CollectorResult> writeIMUDataToFile(CollectorResult result, File saveFile) {
        SaveTask task = new SaveTask(saveFile, result, new CompletableFuture<>(), 1);
        pushTask(task);
        return task.getFuture();
    }

    public static void initialize(ScheduledExecutorService service, List<ScheduledFuture<?>> futureList) {
        synchronized (FileSaver.class) {
            if (instance == null) {
                instance = new FileSaver(service, futureList);
            }
        }
    }

    public static FileSaver getInstance() {
        return instance;
    }
}
