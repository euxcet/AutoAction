package com.example.datacollection.data;

import android.content.Context;
import android.os.CountDownTimer;
import android.os.Handler;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;

import com.example.datacollection.BuildConfig;
import com.example.datacollection.utils.RandomUtils;
import com.example.datacollection.utils.bean.TaskListBean;
import com.example.datacollection.utils.FileUtils;
import com.example.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

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
    private File sensorFile;
    private File sensorBinFile;
    private File cameraFile;
    private File microphoneFile;
    private File timestampFile;

    private TaskListBean taskList;
    private TaskListBean.Task task;
    private TaskListBean.Task.Subtask subtask;
    private CountDownTimer timer;
    private RecorderListener listener;

    private String recordId;

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

    public void start(String name, int taskId, int subtaskId, TaskListBean taskList) {
        this.taskList = taskList;
        this.task = taskList.getTask().get(taskId);
        this.subtask = task.getSubtask().get(subtaskId);
        this.tickCount = 0;
        this.recordId = RandomUtils.generateRandomRecordId();

        if (subtask.getTimes() == 0) {
            subtask.setTimes(task.getTimes());
        }
        if (subtask.getDuration() == 0) {
            subtask.setDuration(task.getDuration());
        }
        subtask.setAudio(subtask.isAudio() | task.isAudio());
        subtask.setVideo(subtask.isVideo() | task.isVideo());

        createFile(name, taskId, subtaskId);

        long duration = subtask.getDuration();
        int times = subtask.getTimes();
        long actionTime = times * subtask.getDuration();

        timer = new CountDownTimer(actionTime, duration) {
            @Override
            public void onTick(long l) {
                if (l < duration / 10) { // skip first tick
                    return;
                }
                timestampController.add(sensorController.getLastTimestamp());
                tickCount += 1;
                listener.onTick(tickCount, times);
            }

            @Override
            public void onFinish() {
                listener.onFinish();
                stop();
            }
        };

        new Handler().postDelayed(() -> {
            sensorController.start(sensorFile, sensorBinFile);
            /*
            if (subtask.isAudio()) {
                microphoneController.start(microphoneFile);
            }
             */
            if (subtask.isVideo()) {
                cameraController.start(cameraFile);
            }
            timestampController.start(timestampFile);
            timer.start();
        }, 3000);
    }

    public void interrupt() {
        timer.cancel();
        sensorController.stop();
        /*
        if (subtask != null && subtask.isAudio()) {
            microphoneController.stop();
        }
         */
        if (subtask != null && subtask.isVideo()) {
            cameraController.stop();
        }
        timestampController.stop();
    }

    private void stop() {
        sensorController.stop();
        /*
        if (subtask != null && subtask.isAudio()) {
            microphoneController.stop();
        }
         */
        if (subtask != null && subtask.isVideo()) {
            cameraController.stop();
        }
        timestampController.stop();

        NetworkUtils.addRecord(mContext, taskList.getId(), task.getId(), subtask.getId(), recordId, System.currentTimeMillis(), new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {

            }
        });
        new Handler().postDelayed(() -> {
            long timestamp = System.currentTimeMillis();
            sensorController.upload(taskList.getId(), task.getId(), subtask.getId(), recordId, timestamp);
            /*
            if (subtask != null && subtask.isAudio()) {
                microphoneController.upload(taskList.getId(), task.getId(), subtask.getId(), recordId, System.currentTimeMillis());
            }
             */
            if (subtask != null && subtask.isVideo()) {
                cameraController.upload(taskList.getId(), task.getId(), subtask.getId(), recordId, timestamp);
            }
            timestampController.upload(taskList.getId(), task.getId(), subtask.getId(), recordId, timestamp);
        }, 3000);
    }

    public void createFile(String name, int taskId, int subtaskId) {
        String suffix = name + "_" + taskId + "_" + subtaskId + "_" + dateFormat.format(new Date());
        timestampFile = new File(BuildConfig.SAVE_PATH, "Timestamp_" + suffix + ".json");
        sensorFile = new File(BuildConfig.SAVE_PATH, "Sensor_" + suffix + ".json");
        microphoneFile = new File(BuildConfig.SAVE_PATH, "Microphone_" + suffix + ".mp4");
        cameraFile = new File(BuildConfig.SAVE_PATH, "Camera_" + suffix + ".mp4");
        sensorBinFile = new File(BuildConfig.SAVE_PATH, "SensorBin_" + suffix + ".bin");
    }

    public interface RecorderListener {
        void onTick(int tickCount, int times);
        void onFinish();
    }
}
