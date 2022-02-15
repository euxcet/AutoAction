package com.example.datacollection;

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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.StringWriter;
import java.io.Writer;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.EasyPermissions;
import pub.devrel.easypermissions.PermissionRequest;

public class MainActivity extends AppCompatActivity {

    public static Context mContext;
    private static TransferData transferData;
    private FilenameFormat filenameFormat;
    private Vibrator vibrator;

    // ui
    private EditText user;
    private Button startButton;
    private Button stopButton;
    private TextView description;
    private TextView counter;
    private CountDownTimer idxTimer;

    private Spinner taskSpinner;
    private ArrayAdapter<String> taskAdapter;

    private Spinner subtaskSpinner;
    private ArrayAdapter<String> subtaskAdapter;

    // task
    private TaskList taskList;
    private String[] tasks = new String[6];
    private String[] taskName;
    private String[] subtaskName;
    private int curTask = 0;
    private int curSubtask = 0;

    private CheckBox cameraSwitch;


    // each action takes 3s = 3000ms
    private int interval = 3000;
    private int repeatTimes = 5;
    private int actionTime = interval * repeatTimes;
    private int curIdx = 0;

    // save file path
    private String pathName = "/storage/emulated/0/PlaceData/";

    // sensor
    private SensorManager sensorManager;
    private int samplingPeriod = SensorManager.SENSOR_DELAY_FASTEST;  // fastest
    private Sensor gyroSensor;
    private Sensor linearAccSensor;
    private Sensor accSensor;
    private Sensor magSensor;
    private boolean sensorSupported = true;

    // microphone
    private File file, audioFile, videoFile;
    private MediaRecorder mMediaRecorder;

    // camera
    private CameraController cameraController;
    private boolean isVideo;


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
        filenameFormat = FilenameFormat.getInstance();
        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        // initTasks();
        taskList = TaskList.parseFromFile(getResources().openRawResource(R.raw.tasklist));
        initView();

        // camera
        cameraController = new CameraController(this);

        filenameFormat.setPathName(pathName);

        // sensor
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        sensorManager.registerListener(gyroListener, gyroSensor, samplingPeriod);
        linearAccSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        sensorManager.registerListener(linearAccListener, linearAccSensor, samplingPeriod);
        accSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensorManager.registerListener(accListener, accSensor, samplingPeriod);
        magSensor = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        sensorManager.registerListener(magListener, magSensor, samplingPeriod);

        if (!isSensorSupport())
            sensorSupported = false;
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

    private boolean isSensorSupport() {
        return gyroSensor != null && linearAccSensor != null && accSensor != null && magSensor != null;
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
                curTask = position;
                subtaskName = taskList.getTask().get(curTask).getSubtaskName();
                subtaskAdapter = new ArrayAdapter<>(mContext, android.R.layout.simple_spinner_item, subtaskName);
                subtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                subtaskSpinner.setAdapter(subtaskAdapter);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        subtaskName = taskList.getTask().get(curTask).getSubtaskName();
        subtaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, subtaskName);
        subtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        subtaskSpinner.setAdapter(subtaskAdapter);
        subtaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                curSubtask = position;
                description.setText(subtaskName[curSubtask]);
                interval = taskList.getTask().get(curTask).getSubtask().get(curSubtask).getDuration();
                repeatTimes = taskList.getTask().get(curTask).getSubtask().get(curSubtask).getTimes();
                actionTime = interval * repeatTimes;
                isVideo = taskList.getTask().get(curTask).getSubtask().get(curSubtask).isVideo();
                cameraSwitch.setChecked(isVideo);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        cameraSwitch = findViewById(R.id.video_switch);
        cameraSwitch.setOnCheckedChangeListener((compoundButton, b) -> {
            if (b) {
                cameraController.openCamera();
            } else {
                cameraController.closeCamera();
            }
        });
        cameraSwitch.setEnabled(false);

        startButton = findViewById(R.id.start);
        stopButton = findViewById(R.id.stop);

        startButton.setOnClickListener(view -> {
            if (!sensorSupported) {
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

        enableButtons(false);
    }

    private void startAudioRecording() {
        if (mMediaRecorder != null) {
            mMediaRecorder.start();
        }
    }

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

    private void enableButtons(boolean isRecording) {
        startButton.setEnabled(!isRecording);
        stopButton.setEnabled(isRecording);
    }

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

    private static void makeRootDirectory(String filePath) {
        try {
            File file = new File(filePath);
            if (!file.exists())
                file.mkdir();
        } catch (Exception e) {
            Log.e("error:", e + "");
        }
    }

    // sensor
    private SensorEventListener gyroListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            float gyrox = event.values[0];
            float gyroy = event.values[1];
            float gyroz = event.values[2];
            SensorInfo info = new SensorInfo(Sensor.TYPE_GYROSCOPE, gyrox, gyroy, gyroz, event.timestamp);
            transferData.addSensorData(info);
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    };

    private SensorEventListener linearAccListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            float linearx = event.values[0];
            float lineary = event.values[1];
            float linearz = event.values[2];
            SensorInfo info = new SensorInfo(Sensor.TYPE_LINEAR_ACCELERATION, linearx, lineary, linearz, event.timestamp);
            transferData.addSensorData(info);
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    };

    private SensorEventListener accListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            float accx = event.values[0];
            float accy = event.values[1];
            float accz = event.values[2];
            SensorInfo info = new SensorInfo(Sensor.TYPE_ACCELEROMETER, accx, accy, accz, event.timestamp);
            transferData.addSensorData(info);
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    };

    private SensorEventListener magListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            float magx = event.values[0];
            float magy = event.values[1];
            float magz = event.values[2];
            SensorInfo info = new SensorInfo(Sensor.TYPE_MAGNETIC_FIELD, magx, magy, magz, event.timestamp);
            transferData.addSensorData(info);
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {

        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (sensorManager != null) {
            sensorManager.unregisterListener(gyroListener);
            sensorManager.unregisterListener(linearAccListener);
            sensorManager.unregisterListener(accListener);
            sensorManager.unregisterListener(magListener);
        }
        if (vibrator != null)
            vibrator.cancel();
    }
}
