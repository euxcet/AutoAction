package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaRecorder;
import android.media.MediaScannerConnection;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.TransferData;
import com.example.datacollection.data.CameraController;
import com.example.datacollection.data.Recorder;
import com.example.datacollection.data.SensorController;

import java.io.File;
import java.io.IOException;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.EasyPermissions;
import pub.devrel.easypermissions.PermissionRequest;

public class MainActivity extends AppCompatActivity {

    public static Context mContext;
    private static TransferData transferData;
    private Vibrator vibrator;

    // ui
    private EditText user;
    private Button startButton;
    private Button stopButton;
    private TextView description;
    private TextView counter;

    private Spinner taskSpinner;
    private ArrayAdapter<String> taskAdapter;

    private Spinner subtaskSpinner;
    private ArrayAdapter<String> subtaskAdapter;

    // task
    private TaskList taskList;
    private String[] taskName;
    private String[] subtaskName;
    private int curTaskId = 0;
    private int curSubtaskId = 0;
    private TaskList.Task.Subtask curTask;

    private boolean isVideo;

    private CheckBox cameraSwitch;

    // each action takes 3s = 3000ms
    /*
    private int interval = 3000;
    private int repeatTimes = 5;
    private int actionTime = interval * repeatTimes;
     */

    // save file path
    private String pathName = "/storage/emulated/0/PlaceData/";

    private Recorder recorder;


    // permission
    private static final int RC_PERMISSIONS = 0;
    private String[] permissions = new String[]{
            Manifest.permission.INTERNET,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA,
            Manifest.permission.VIBRATE,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        requestPermissions();

        mContext = this;

        transferData = TransferData.getInstance();
        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        taskList = TaskList.parseFromFile(getResources().openRawResource(R.raw.tasklist));
        initView();

        vibrator = (Vibrator) mContext.getSystemService(Context.VIBRATOR_SERVICE);

        recorder = new Recorder(this, pathName, new Recorder.RecorderListener() {
            @Override
            public void onTick(int tickCount) {
                vibrator.vibrate(VibrationEffect.createOneShot(200, 128));
            }

            @Override
            public void onFinish() {
                vibrator.vibrate(VibrationEffect.createOneShot(600, 128));
            }
        });
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        // Forward results to EasyPermissions
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
    }

    @AfterPermissionGranted(RC_PERMISSIONS)
    private void requestPermissions() {
        if (EasyPermissions.hasPermissions(this, permissions)) {
            // have permissions
        } else {
            // no permissions, request dynamically
            EasyPermissions.requestPermissions(
                    new PermissionRequest.Builder(this, RC_PERMISSIONS, permissions)
                            .setRationale(R.string.rationale)
                            .setPositiveButtonText(R.string.rationale_ask_ok)
                            .setNegativeButtonText(R.string.rationale_ask_cancel)
                            .setTheme(R.style.Theme_AppCompat)
                            .build());
        }
    }


    private void initView() {
        user = findViewById(R.id.user);
        user.setText("aa");
        description = findViewById(R.id.description);
        counter = findViewById(R.id.counter);

        // Spinner
        taskSpinner = findViewById(R.id.task_spinner);
        subtaskSpinner = findViewById(R.id.subtask_spinner);

        taskName = taskList.getTaskName();
        taskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, taskName);
        taskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        taskSpinner.setAdapter(taskAdapter);
        taskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                curTaskId = position;
                subtaskName = taskList.getTask().get(curTaskId).getSubtaskName();
                subtaskAdapter = new ArrayAdapter<>(mContext, android.R.layout.simple_spinner_item, subtaskName);
                subtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                subtaskSpinner.setAdapter(subtaskAdapter);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        subtaskName = taskList.getTask().get(curTaskId).getSubtaskName();
        subtaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, subtaskName);
        subtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        subtaskSpinner.setAdapter(subtaskAdapter);
        subtaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                curSubtaskId = position;
                description.setText(subtaskName[curSubtaskId]);
                curTask = taskList.getTask().get(curTaskId).getSubtask().get(curSubtaskId);
                isVideo = taskList.getTask().get(curTaskId).getSubtask().get(curSubtaskId).isVideo();
                cameraSwitch.setChecked(isVideo);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        cameraSwitch = findViewById(R.id.video_switch);
        cameraSwitch.setOnCheckedChangeListener((compoundButton, b) -> {
            recorder.setCamera(b);
        });
        cameraSwitch.setEnabled(false);

        startButton = findViewById(R.id.start);
        stopButton = findViewById(R.id.stop);

        startButton.setOnClickListener(view -> {
            enableButtons(true);
            recorder.start(
                    user.getText().toString(),
                    curTaskId,
                    curSubtaskId,
                    curTask
            );
        });

        stopButton.setOnClickListener(view -> {
            recorder.stop();
            enableButtons(false);
        });

        enableButtons(false);
    }

    private void enableButtons(boolean isRecording) {
        startButton.setEnabled(!isRecording);
        stopButton.setEnabled(isRecording);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (vibrator != null) {
            vibrator.cancel();
        }
    }
}

            /*
        startButton.setOnClickListener(view -> {
            if (!sensorController.isSensorSupport()) {
                Toast.makeText(mContext, "传感器缺失", Toast.LENGTH_LONG).show();
                return;
            }

            if (user.getText().toString().length() == 0 || curTask == -1) {
                Toast.makeText(mContext, "请填写用户名并选择任务", Toast.LENGTH_LONG).show();
                return;
            }

            filenameFormat.setFilename(user.getText().toString(), String.valueOf(curTask), String.valueOf(curSubtask));

            // save time stamp
            curIdx = 0;

            idxTimer = new CountDownTimer(actionTime, interval) {
                @Override
                public void onTick(long l) {
                    if (l <= interval / 10)
                        return;
                    transferData.addTimestampData();
                    counter.setText(String.format("%d/%d", curIdx, repeatTimes));

                    if (isVideo) {
                        if (curIdx > 0) {
                            cameraController.stopRecording();
                            cameraController.closeCamera();
                        }

                        new Handler().postDelayed(() -> {
                            Log.e("TAG", "vibrate");
                            vibrator.vibrate(VibrationEffect.createOneShot(200, 128));
                            cameraController.openCamera();
                        },300);
                        new Handler().postDelayed(() -> {
                            cameraController.startRecording(
                                    new File(filenameFormat.getPathName(),
                                            filenameFormat.getVideoFilename(curIdx) + ".mp4")
                            );
                            curIdx++;
                        }, 600);

                    } else {
                        vibrator.vibrate(VibrationEffect.createOneShot(200, 128));
                        curIdx++;
                    }
                }

                @Override
                public void onFinish() {
                    if (isVideo) {
                        cameraController.stopRecording();
                        cameraController.closeCamera();
                    }
                    done();
                    counter.setText(String.format("%d/%d", repeatTimes, repeatTimes));
                    vibrator.vibrate(VibrationEffect.createOneShot(600, 128));
                }
            };

            // microphone
            setupMediaRecorder();

            // start
            enableButtons(true);
            counter.setText("");

            if (isVideo) {
                cameraController.closeCamera();
            }

            new Handler().postDelayed(() -> {
                transferData.startRecording();
                startAudioRecording();
                idxTimer.start();
            }, 3000);
        });

        stopButton.setOnClickListener(view -> {
            idxTimer.cancel();
            stop();
        });
             */
    /*
    // microphone
    private void createDataFile() {
        makeRootDirectory(filenameFormat.getPathName());
        String fileName = filenameFormat.getMicrophoneFilename();
        file = new File(filenameFormat.getPathName(), fileName + ".txt");
        try {
            if (!file.exists()) {
                file.createNewFile();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        audioFile = new File(filenameFormat.getPathName(), fileName + ".mp4");
    }

    @SuppressLint({"MissingPermission", "CheckResult"})
    private void setupMediaRecorder() {
        try {
            createDataFile();
            mMediaRecorder = new MediaRecorder();
            mMediaRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            mMediaRecorder.setAudioChannels(2);
            mMediaRecorder.setAudioSamplingRate(44100);
            mMediaRecorder.setAudioEncodingBitRate(16 * 44100);
            mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            mMediaRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            mMediaRecorder.setOutputFile(audioFile);
            mMediaRecorder.prepare();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
     */
    /*
    private void stopRecording() {
        if (mMediaRecorder != null) {
            mMediaRecorder.stop();
            mMediaRecorder.release();
            mMediaRecorder = null;
            MediaScannerConnection.scanFile(mContext,
                    new String[] {file.getAbsolutePath(), audioFile.getAbsolutePath()},
                    null, null);
        }
    }
     */

    /*
    private void stop() {
        stopRecording();
        transferData.stopRecording();

        // clear data
        transferData.clear();
        counter.setText("");

        enableButtons(false);
    }

    private void done() {
        stopRecording();
        transferData.stopRecording();

        // upload data
        transferData.upload(mContext);
        counter.setText("");

        enableButtons(false);
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
     */
