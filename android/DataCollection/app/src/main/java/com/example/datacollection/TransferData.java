package com.example.datacollection;

import android.content.Context;
import android.util.Log;
import android.widget.Toast;

import com.google.gson.Gson;
import com.lzy.okgo.OkGo;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;

public class TransferData {

    private boolean isRecording = false;

//    private String serverAddr = "http://183.173.98.59:8888";

    private long lastTimestamp;
    private List<Long> timestampData = new ArrayList<>();
    private List<SensorInfo> sensorData = new ArrayList<>();

    private FilenameFormat filenameFormat = FilenameFormat.getInstance();

    private static TransferData transferData;

    public static TransferData getInstance() {
        if (transferData == null)
            transferData = new TransferData();
        return transferData;
    }

    public void addSensorData(SensorInfo info) {
        if (isRecording) {
            sensorData.add(info);
            lastTimestamp = info.getTime();
        }
    }

    public void addTimestampData() {
        if (isRecording)
            timestampData.add(lastTimestamp);
    }

    public void startRecording() {
        isRecording = true;
    }

    public void stopRecording() {
        isRecording = false;
    }

    public void upload(final Context context) {
        Gson gson = new Gson();
//        String url = serverAddr + "/upload";

        // sensor
        String sensorToJson = gson.toJson(sensorData);
        String sensorFileName = filenameFormat.getSensorFilename() + ".json";
        writeTxtToFile(sensorToJson, filenameFormat.getPathName(), sensorFileName);
        File sensorFile = new File(filenameFormat.getPathName(), sensorFileName);
        Log.e("json", "sensor file size: " + sensorFile.length());
        if (sensorFile.length() < 500 * 1e3)  // < 500KB
            Toast.makeText(context, String.format("数据量异常：%.2fMB", 1.0 * sensorFile.length() / 1e6), Toast.LENGTH_LONG).show();
//        postFile(context, url, sensorFile, "Sensor Data");

        // timestamp
        String timestampToJson = gson.toJson(timestampData);
        String timestampFileName = filenameFormat.getTimestampFilename() + ".json";
        writeTxtToFile(timestampToJson, filenameFormat.getPathName(), timestampFileName);
        File timestampFile = new File(filenameFormat.getPathName(), timestampFileName);
        Log.e("json", "timestamp file size: " + timestampFile.length());
//        postFile(context, url, timestampFile, "Timestamp Data");

        // microphone
        String microphoneFileName = filenameFormat.getMicrophoneFilename() + ".mp4";
        File microphoneFile = new File(filenameFormat.getPathName(), microphoneFileName);
        Log.e("json", "microphone file size: " + microphoneFile.length());
//        postFile(context, url, microphoneFile, "Microphone Data");

        clear();
    }

    public void clear() {
        sensorData.clear();
        timestampData.clear();
    }

    private static void writeTxtToFile(String content, String filePath, String fileName) {
        makeFilePath(filePath, fileName);
        String strFilePath = filePath + fileName;
        String toWrite = content + "\r\n";
        try {
            File file = new File(strFilePath);
            if (!file.exists()) {
                Log.e("TestFile", "Create the file:" + strFilePath);
                file.getParentFile().mkdirs();
                file.createNewFile();
            } else {
                file.delete();
                file.createNewFile();
            }
            RandomAccessFile raf = new RandomAccessFile(file, "rwd");
            raf.seek(0);
            raf.write(toWrite.getBytes());
            raf.close();
        } catch (Exception e) {
            Log.e("TestFile", "Error on write File:" + e);
        }
    }

    private void postFile(Context context, String url, final File file, final String tag) {
        OkGo.<String>post(url)
                .tag(context)
                .params("file", file)
                .isMultipart(true)
                .execute(new StringCallback() {
                    @Override
                    public void onSuccess(Response<String> response) {
                        Log.e("Upload", tag + " Upload Succeeded.");
//                        file.delete();
                    }

                    @Override
                    public void onError(Response<String> response) {
                        Log.e("Upload", tag + " Upload failed.");
                    }
                });
    }

    private static File makeFilePath(String filePath, String fileName) {
        File file = null;
        makeRootDirectory(filePath);
        try {
            file = new File(filePath + fileName);
            if (!file.exists())
                file.createNewFile();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return file;
    }

    private static void makeRootDirectory(String filePath) {
        try {
            File file = new File(filePath);
            if (!file.exists())
                file.mkdir();
        } catch (Exception e) {
            Log.e("error:", e + "");
        }
    }
}
