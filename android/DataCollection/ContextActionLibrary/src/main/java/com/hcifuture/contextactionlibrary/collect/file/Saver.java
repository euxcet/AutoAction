package com.hcifuture.contextactionlibrary.collect.file;

import android.content.Context;
import android.os.Build;
import android.text.TextUtils;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Saver {
    private static final String TAG = "Saver";

    private String saveFolder;
    private String savePath;
    private ScheduledExecutorService scheduledExecutorService;
    private List<ScheduledFuture<?>> futureList;

    public Saver(Context context, String triggerFolder, String saveFolderName, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        this.saveFolder = context.getExternalMediaDirs()[0].getAbsolutePath() + "/" + triggerFolder + "/" + saveFolderName;
        this.savePath = "";
        this.scheduledExecutorService = scheduledExecutorService;
        this.futureList = futureList;
    }

    private void saveString(String data) {
        try {
            File file = new File(savePath);
            if (!Objects.requireNonNull(file.getParentFile()).exists()) {
                file.getParentFile().mkdirs();
            }
            FileOutputStream fos = new FileOutputStream(file);
            OutputStreamWriter writer = new OutputStreamWriter(fos, StandardCharsets.UTF_8);
            writer.write(data);
            writer.close();
            fos.flush();
            fos.close();
        } catch (IOException e) {
            Log.e("Exception", "File write failed:" + e.toString());
        }
    }

    private void saveFloatList(List<Float> data) {
        try {
            File file = new File(savePath);
            if (!Objects.requireNonNull(file.getParentFile()).exists()) {
                file.getParentFile().mkdirs();
            }
            FileOutputStream fos = new FileOutputStream(file);
            DataOutputStream dos = new DataOutputStream(fos);
            for (Float value: data) {
                dos.writeFloat(value);
            }
            dos.flush();
            dos.close();
            fos.flush();
            fos.close();
        } catch (IOException e) {
            Log.e("Exception", "File write failed:" + e.toString());
        }

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public CompletableFuture<Void> save(Object object) {
        Log.e("TapTapCollector", "in save");
        if (object == null) {
            return CompletableFuture.completedFuture(null);
        }
        CompletableFuture<Void> ft = new CompletableFuture<>();
        if (object instanceof List) {
            futureList.add(scheduledExecutorService.schedule(() -> {
                try {
                    saveFloatList((List<Float>) object);
                    ft.complete(null);
                } catch (Exception exc) {
                    ft.completeExceptionally(exc);
                }
            }, 0, TimeUnit.MILLISECONDS));
        } else {
            futureList.add(scheduledExecutorService.schedule(() -> {
                try {
                    if (object instanceof String) {
                        saveString((String)object);
                    } else {
                        saveString(new GsonBuilder().disableHtmlEscaping().create().toJson(object));
                    }
                    ft.complete(null);
                } catch (Exception exc) {
                    ft.completeExceptionally(exc);
                }
            }, 0, TimeUnit.MILLISECONDS));
        }
        return ft;
    }

    public void setSavePath(String label) {
        savePath = this.saveFolder + "/" + label;
    }

    public String getSavePath() {
        return savePath;
    }

    public String getSaveFolder() {
        return saveFolder;
    }

    public void deleteFolderFile(String filePath, boolean shouldDelete) {
        if (!TextUtils.isEmpty(filePath)) {
            try {
                File file = new File(filePath);
                if (file.isDirectory()) {
                    File[] files = file.listFiles();
                    for (File file0 : files)
                        deleteFolderFile(file0.getAbsolutePath(), true);
                }
                if (shouldDelete) {
                    Log.e("TapTapCollector delete", file.getAbsolutePath());
                    if (!file.isDirectory()) {
                        file.delete();
                    } else {
                        if (file.listFiles().length == 0) {
                            file.delete();
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
