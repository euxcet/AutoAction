package com.example.contextactionlibrary.collect.file;

import android.content.Context;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.google.gson.Gson;

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
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Saver {
    private static final String TAG = "Saver";

    private String saveFolder;
    private String savePath;
    private Executor pool;

    public Saver(Context context, String triggerFolder, String saveFolderName) {
        this.saveFolder = context.getExternalMediaDirs()[0].getAbsolutePath() + "/" + triggerFolder + "/" + saveFolderName;
        this.savePath = "";
        this.pool = new ThreadPoolExecutor(1, 1,
                60, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(10),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor.DiscardPolicy());
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
        if (object == null) {
            return CompletableFuture.completedFuture(null);
        }
        CompletableFuture<Void> ft = new CompletableFuture<>();
        if (object instanceof List) {
            pool.execute(() -> {
                try {
                    saveFloatList((List<Float>) object);
                    ft.complete(null);
                } catch (Exception exc) {
                    ft.completeExceptionally(exc);
                }
            });
        } else {
            pool.execute(() -> {
                try {
                    saveString(new Gson().toJson(object));
                    ft.complete(null);
                } catch (Exception exc) {
                    ft.completeExceptionally(exc);
                }
            });
        }
        return ft;
    }

    public void setSavePath(String label) {
        savePath = this.saveFolder + "/" + label;
    }

    public String getSavePath() {
        return savePath;
    }
}
