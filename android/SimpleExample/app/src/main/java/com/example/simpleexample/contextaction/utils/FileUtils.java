package com.example.simpleexample.contextaction.utils;

import static android.content.Context.MODE_PRIVATE;

import android.content.Context;
import android.content.SharedPreferences;

import com.example.simpleexample.BuildConfig;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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

    public static void copy(File src, File dst) {
        makeDir(dst.getParent());
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

    public interface DownloadListener {
        void onFinished();
    }

    public interface CheckListener {
        void onChanged(List<String> changedFilename, List<String> serverMD5s);
    }

    public static void downloadFiles(Context context, List<String> filename, DownloadListener listener) {
        if (filename.isEmpty()) {
            listener.onFinished();
            return;
        }
        AtomicInteger counter = new AtomicInteger(filename.size());
        for (String name: filename) {
            NetworkUtils.downloadFile(context, name, new FileCallback() {
                @Override
                public void onSuccess(Response<File> response) {
                    File file = response.body();
                    File saveFile = new File(BuildConfig.SAVE_PATH, name);
                    FileUtils.copy(file, saveFile);
                    file.delete();
                    if (counter.decrementAndGet() == 0) {
                        listener.onFinished();
                    }
                }
            });
        }
    }

    public static void checkFiles(Context context, List<String> filename, CheckListener listener) {
        if (filename.isEmpty()) {
            listener.onChanged(new ArrayList<>(), new ArrayList<>());
            return;
        }
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
                if (md5s.length != filename.size()) {
                    return;
                }
                for (int i = 0; i < filename.size(); i++) {
                    String serverMD5 = md5s[i];
                    String localMD5 = fileMD5.getString(filename.get(i), null);
                    if (localMD5 == null || !localMD5.equals(serverMD5)) {
                        changedFilename.add(filename.get(i));
                    }
                }
                listener.onChanged(changedFilename, Arrays.asList(md5s));
            }
        });
    }

    public static String fileToMD5(String path) {
        try {
            InputStream inputStream = new FileInputStream(path);
            byte[] buffer = new byte[1024];
            MessageDigest digest = MessageDigest.getInstance("MD5");
            int numRead = 0;
            while (numRead != -1) {
                numRead = inputStream.read(buffer);
                if (numRead > 0)
                    digest.update(buffer, 0, numRead);
            }
            byte [] md5Bytes = digest.digest();
            return convertHashToString(md5Bytes);
        } catch (Exception e) {
            return "";
        }
    }

    private static String convertHashToString(byte[] hashBytes) {
        StringBuilder returnVal = new StringBuilder();
        for (int i = 0; i < hashBytes.length; i++) {
            returnVal.append(Integer.toString((hashBytes[i] & 0xff) + 0x100, 16).substring(1));
        }
        return returnVal.toString().toLowerCase();
    }
}
