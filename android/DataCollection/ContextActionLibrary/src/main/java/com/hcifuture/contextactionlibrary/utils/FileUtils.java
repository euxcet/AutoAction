package com.hcifuture.contextactionlibrary.utils;

import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.sensor.collector.CollectorResult;
import com.hcifuture.contextactionlibrary.sensor.data.IMUData;
import com.hcifuture.contextactionlibrary.sensor.data.SingleIMUData;

import java.io.BufferedReader;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

public class FileUtils {

    public static void makeDir(String directory) {
        try {
            File file = new File(directory);
            if (!file.exists()) {
                file.mkdirs();
            }
        } catch (Exception ignored) {
        }
    }

    public static File makeFile(File file) {
        try {
            makeDir(file.getParent());
            if (!file.exists()) {
                file.createNewFile();
            }
        } catch (Exception ignored) {
        }
        return file;
    }

    public static void writeStringToFile(String content, File saveFile) {
        makeFile(saveFile);
        String toWrite = content + "\r\n";
        try {
            OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(saveFile));
            writer.write(toWrite);
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public static CompletableFuture<CollectorResult> writeStringToFile(CollectorResult result, File saveFile, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        futureList.add(scheduledExecutorService.schedule(() -> {
            try {
                makeFile(saveFile);
                String toWrite = result.getDataString() + "\r\n";
                OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(saveFile));
                writer.write(toWrite);
                writer.close();
                result.setSavePath(saveFile.getAbsolutePath());
                ft.complete(result);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, 0, TimeUnit.MILLISECONDS));
        return ft;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public static CompletableFuture<CollectorResult> writeIMUDataToFile(CollectorResult result, File saveFile, ScheduledExecutorService scheduledExecutorService, List<ScheduledFuture<?>> futureList) {
        assert result.getData() instanceof IMUData;
        CompletableFuture<CollectorResult> ft = new CompletableFuture<>();
        futureList.add(scheduledExecutorService.schedule(() -> {
            try {
                makeFile(saveFile);
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
        }, 0, TimeUnit.MILLISECONDS));
        return ft;
    }

    public static void copy(File src, File dst) {
        try {
            InputStream in = new FileInputStream(src);
            try {
                OutputStream out = new FileOutputStream(dst);
                try {
                    byte[] buf = new byte[1024];
                    int len;
                    while ((len = in.read(buf)) > 0) {
                        out.write(buf, 0, len);
                    }
                } finally {
                    out.close();
                }
            } finally {
                in.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String getFileContent(String filename) {
        Log.e("Uploader", "Get file content " + filename);
        StringBuffer buffer = new StringBuffer();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new FileReader(filename));
            String line;
            while ((line = reader.readLine()) != null) {
                buffer.append(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (reader != null) {
                    reader.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return buffer.toString();
    }

    public static void deleteFile(File file, String tag) {
        long threadId = Thread.currentThread().getId();
        if (file.delete()) {
            Log.d("FileUtils"+threadId, "[" + tag + "] Delete " + file.getAbsolutePath() + " successfully");
        } else {
            Log.d("FileUtils"+threadId, "[" + tag + "] Failed to delete " + file.getAbsolutePath());
        }
    }
}
