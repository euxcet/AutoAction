package com.example.datacollection.data;

import android.content.Context;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.Vibrator;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.example.datacollection.BuildConfig;
import com.example.datacollection.TaskList;
import com.example.datacollection.utils.FileUtils;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Recorder {

    private Context mContext;

    private MicrophoneController microphoneController;
    private CameraController cameraController;
    private SensorController sensorController;
    private TimestampController timestampController;

    // file
    private File cameraFile;
    private File sensorFile;
    private File microphoneFile;
    private File timestampFile;

    private TaskList.Task.Subtask subtask;
    private CountDownTimer timer;
    private RecorderListener listener;

    private int tickCount = 0;

    private final SimpleDateFormat dateFormat = new SimpleDateFormat("yyMMddHHmmss");

    public Recorder(Context context, RecorderListener listener) {
        this.mContext = context;
        this.listener = listener;
        cameraController = new CameraController((AppCompatActivity) mContext);
        sensorController = new SensorController(mContext);
        microphoneController = new MicrophoneController(mContext);
        timestampController = new TimestampController(mContext);
        FileUtils.makeDir(BuildConfig.SAVE_PATH);
    }

    public void setCamera(boolean b) {
        if (b) {
            cameraController.openCamera();
        } else {
            cameraController.closeCamera();
        }
    }

    public void start(String name, int taskId, int subtaskId, TaskList.Task.Subtask subtask) {
        this.subtask = subtask;
        this.tickCount = 0;

        createFile(name, taskId, subtaskId);

        long duration = subtask.getDuration();
        long actionTime = subtask.getTimes() * subtask.getDuration();

        timer = new CountDownTimer(actionTime, duration) {
            @Override
            public void onTick(long l) {
                if (l < duration / 10) { // skip first tick
                    return;
                }
                timestampController.add(sensorController.getLastTimestamp());
                tickCount += 1;
                listener.onTick(tickCount);
            }

            @Override
            public void onFinish() {
                listener.onFinish();
                stop();
            }
        };

        new Handler().postDelayed(() -> {
            sensorController.start(sensorFile);
            if (subtask.isAudio()) {
                microphoneController.start(microphoneFile);
            }
            if (subtask.isVideo()) {
                cameraController.start(cameraFile);
            }
            timestampController.start(timestampFile);
        }, 3000);
    }

    public void stop() {
        sensorController.stop();
        if (subtask != null && subtask.isAudio()) {
            microphoneController.stop();
        }
        if (subtask != null && subtask.isVideo()) {
            cameraController.stop();
        }
        timestampController.stop();
        new Handler().postDelayed(() -> {
            sensorController.upload();
            if (subtask != null && subtask.isAudio()) {
                microphoneController.upload();
            }
            if (subtask != null && subtask.isVideo()) {
                cameraController.upload();
            }
            timestampController.upload();
        }, 3000);
    }

    public void createFile(String name, int taskId, int subtaskId) {
        String suffix = "_" + name + "_" + taskId + "_" + subtaskId + "_" + dateFormat.format(new Date());
        timestampFile = new File(BuildConfig.SAVE_PATH, "Timestamp" + suffix + ".txt");
        sensorFile = new File(BuildConfig.SAVE_PATH, "Sensor" + suffix + ".json");
        microphoneFile = new File(BuildConfig.SAVE_PATH, "Microphone" + suffix + ".mp4");
        cameraFile = new File(BuildConfig.SAVE_PATH, "Camera" + suffix + ".mp4");
    }

    public interface RecorderListener {
        void onTick(int tickCount);
        void onFinish();
    }
}
