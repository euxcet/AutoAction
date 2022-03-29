package com.hcifuture.datacollection.utils;

import static android.content.Context.MODE_PRIVATE;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.NcnnInstance;
import com.hcifuture.datacollection.data.SensorInfo;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class FileUtils {

    public static void makeDir(String directory) {
        try {
            File file = new File(directory);
            if (!file.exists()) {
                file.mkdir();
            }
        } catch (Exception ignored) {
        }
    }

    private static File makeFile(File file) {
        try {
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

    public static void writeIMUDataToFile(List<SensorInfo> data, File saveFile) {
        makeFile(saveFile);
        try {
            FileOutputStream fos = new FileOutputStream(saveFile);
            DataOutputStream dos = new DataOutputStream(fos);
            for (SensorInfo info: data) {
                for (Float value: info.getData()) {
                    dos.writeFloat(value);
                }
                dos.writeFloat((float)info.getTime());
            }
            dos.flush();
            dos.close();
            fos.flush();
            fos.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
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

    public static List<SensorInfo> loadIMUBinData(File file) {
        List<SensorInfo> data = new ArrayList<>();
        try {
            InputStream fis = new FileInputStream(file);
            DataInputStream dis = new DataInputStream(fis);
            while (dis.available() > 0) {
                float idx = dis.readFloat();
                float x = dis.readFloat();
                float y = dis.readFloat();
                float z = dis.readFloat();
                long timestamp = (long)dis.readFloat();
                data.add(new SensorInfo(idx, x, y, z, timestamp));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public interface DownloadListener {
        void onFinished();
    }

    public interface CheckListener {
         void onChanged(List<String> changedFilename);
    }

    public static void downloadFiles(Context context, List<String> filename, boolean triggerWhenModified, DownloadListener listener) {
        AtomicInteger counter = new AtomicInteger(filename.size());
        AtomicBoolean modified = new AtomicBoolean(false);
        for (String name: filename) {
            NetworkUtils.getMD5(context, name, new StringCallback() {
                @Override
                public void onSuccess(Response<String> response) {
                    String serverMD5 = response.body();
                    SharedPreferences fileMD5 = context.getSharedPreferences("FILE_MD5", MODE_PRIVATE);
                    SharedPreferences.Editor editor = fileMD5.edit();
                    String localMD5 = fileMD5.getString(name, null);
                    if (localMD5 != null && localMD5.equals(serverMD5)) {
                        if (counter.decrementAndGet() == 0) {
                            if (!triggerWhenModified || modified.get()) {
                                listener.onFinished();
                            }
                        }
                        return;
                    }
                    NetworkUtils.downloadFile(context, name, new FileCallback() {
                        @Override
                        public void onSuccess(Response<File> response) {
                            File file = response.body();
                            File saveFile = new File(BuildConfig.SAVE_PATH, name);
                            FileUtils.copy(file, saveFile);
                            editor.putString(name, serverMD5);
                            editor.apply();
                            modified.set(true);
                            if (counter.decrementAndGet() == 0) {
                                if (!triggerWhenModified || modified.get()) {
                                    if (listener != null) {
                                        listener.onFinished();
                                    }
                                }
                            }
                        }
                    });
                }
            });
        }
    }

    public static void checkFiles(Context context, List<String> filename, CheckListener listener) {
        StringBuilder filenameBuilder = new StringBuilder();
        for (String name: filename) {
            filenameBuilder.append(name).append(",");
        }
        NetworkUtils.getMD5(context, filenameBuilder.toString(), new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                SharedPreferences fileMD5 = context.getSharedPreferences("FILE_MD5", MODE_PRIVATE);
                String[] md5s = response.body().split(",");
                List<String> changedFilename = new ArrayList<>();
                Log.e("TEST", md5s.length + " " + filename.size());
                if (md5s.length != filename.size()) {
                    return;
                }
                for (int i = 0; i < filename.size(); i++) {
                    String serverMD5 = md5s[i];
                    String localMD5 = fileMD5.getString(filename.get(i), null);
                    Log.e("TEST", serverMD5 + " " + localMD5);
                    if (localMD5 == null || !localMD5.equals(serverMD5)) {
                        changedFilename.add(filename.get(i));
                    }
                }
                if (!changedFilename.isEmpty()) {
                    listener.onChanged(changedFilename);
                }
            }
        });
    }
}
