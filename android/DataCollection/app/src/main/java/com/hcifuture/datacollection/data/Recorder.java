package com.hcifuture.datacollection.data;

import android.content.Context;
import android.os.CountDownTimer;
import android.os.Handler;

import androidx.appcompat.app.AppCompatActivity;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.utils.RandomUtils;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.utils.FileUtils;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * A very important class for managing sensors, used in MainActivity.
 */
public class Recorder {

    private Context mContext;

    // microphone currently not used
    private MicrophoneController mMicrophoneController;
    private CameraController mCameraController;
    private MotionSensorController mMotionSensorController;
    private LightSensorController mLightSensorController;
    private TimestampController mTimestampController;

    // each of the following file will be passed to the start() function
    // of the corresponding sensor controller and used in it
    private File mMotionSensorFile;
    private File mLightSensorFile;
    private File mCameraFile;
    private File mMicrophoneFile;
    private File mTimestampFile;

    private TaskListBean mTaskList;
    private TaskListBean.Task mTask;
    private TaskListBean.Task.Subtask mSubtask;
    private CountDownTimer mTimer;
    // interface for onTick() and onFinish()
    private RecorderListener mListener;

    private String mRecordId;
    private int mTickCount = 0;
    private final SimpleDateFormat mDateFormat = new SimpleDateFormat("yyMMddHHmmss");
    private String mUserName; // user name set when starting recording

    public Recorder(Context context, RecorderListener listener) {
        this.mContext = context;
        this.mListener = listener;
        mCameraController = new CameraController((AppCompatActivity) mContext);
        mMotionSensorController = new MotionSensorController(mContext);
        mLightSensorController = new LightSensorController(mContext);
        mMicrophoneController = new MicrophoneController(mContext);
        mTimestampController = new TimestampController(mContext);
        mUserName = "DefaultUser";
        FileUtils.makeDir(BuildConfig.SAVE_PATH);
    }

    // open or close the camera
    public void setCamera(boolean b, int lensFacing) {
        if (b) mCameraController.openCamera(lensFacing);
        else mCameraController.closeCamera();
    }

    public void start(String userName, int taskId, int subtaskId, TaskListBean taskList) {
        mTaskList = taskList;
        mTask = taskList.getTasks().get(taskId);
        mSubtask = mTask.getSubtasks().get(subtaskId);
        mTickCount = 0;
        mRecordId = RandomUtils.generateRandomRecordId();
        mUserName = userName;

        createFile(taskId, subtaskId);

        long duration = mSubtask.getDuration();
        int times = mSubtask.getTimes();
        long actionTime = times * mSubtask.getDuration();

        if (mTimer != null) mTimer.cancel();
        mTimer = new CountDownTimer(actionTime, duration) {
            @Override
            public void onTick(long l) {
                if (l < duration / 10) { // skip first tick
                    return;
                }
                mTimestampController.add(mMotionSensorController.getLastTimestamp());
                mTickCount += 1;
                mListener.onTick(mTickCount, times);
            }

            @Override
            public void onFinish() {
                mListener.onFinish();
                stop();
            }
        };

        // canceled Handler().postDelayed(() -> {...});
        mMotionSensorController.start(mMotionSensorFile);
        if (mLightSensorController.isAvailable())
            mLightSensorController.start(mLightSensorFile);
        if (mSubtask.isVideo()) mCameraController.start(mCameraFile);
        if (mSubtask.isAudio()) mMicrophoneController.start(mMicrophoneFile);
        mTimestampController.start(mTimestampFile);
        mTimer.start();
    }

    /**
     * Called when the user want to cancel an ongoing subtask.
     * Cancel all sensors as if this subtask has never been started.
     */
    public void cancel() {
        if (mTimer != null) mTimer.cancel();
        mMotionSensorController.cancel();
        if (mLightSensorController.isAvailable()) mLightSensorController.cancel();
        if (mSubtask != null && mSubtask.isVideo()) mCameraController.cancel();
        if (mSubtask != null && mSubtask.isAudio()) mMicrophoneController.cancel();
        mTimestampController.cancel();
    }

    private void stop() {
        if (mTimer != null) mTimer.cancel();
        mMotionSensorController.stop();
        if (mLightSensorController.isAvailable()) mLightSensorController.stop();
        if (mSubtask.isVideo()) mCameraController.stop();
        if (mSubtask.isAudio()) mMicrophoneController.stop();
        mTimestampController.stop();

        NetworkUtils.addRecord(mContext, mTaskList.getId(), mTask.getId(), mSubtask.getId(),
                mUserName, mRecordId, System.currentTimeMillis(), new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {}
        });

        new Handler().postDelayed(() -> {
            long timestamp = System.currentTimeMillis();
            mMotionSensorController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
            if (mLightSensorController.isAvailable())
                mLightSensorController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
            if (mSubtask.isVideo())
                mCameraController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
            if (mSubtask.isAudio())
                mMicrophoneController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
            mTimestampController.upload(mTaskList.getId(), mTask.getId(), mSubtask.getId(), mRecordId, timestamp);
        }, 2000);
    }

    public void createFile(int taskId, int subtaskId) {
        String suffix = mUserName + "_" + taskId + "_" + subtaskId + "_" + mDateFormat.format(new Date());
        mTimestampFile = new File(BuildConfig.SAVE_PATH, "Timestamp_" + suffix + ".json");
        mMotionSensorFile = new File(BuildConfig.SAVE_PATH, "Motion_" + suffix + ".bin");
        if (mLightSensorController.isAvailable())
            mLightSensorFile = new File(BuildConfig.SAVE_PATH, "Light_" + suffix + ".bin");
        if (mSubtask.isVideo())
            mCameraFile = new File(BuildConfig.SAVE_PATH, "Camera_" + suffix + ".mp4");
        if (mSubtask.isAudio())
            mMicrophoneFile = new File(BuildConfig.SAVE_PATH, "Microphone_" + suffix + ".mp4");
    }

    public interface RecorderListener {
        void onTick(int tickCount, int times);
        void onFinish();
    }
}
