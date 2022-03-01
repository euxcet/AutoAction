package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
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

import com.example.datacollection.BuildConfig;
import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.TransferData;
import com.example.datacollection.data.Recorder;
import com.example.datacollection.utils.FileUtils;
import com.example.datacollection.utils.NetworkUtils;
import com.example.datacollection.utils.bean.StringListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.io.FileInputStream;

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

    private Button configButton;
    private Button trainButton;

    // task
    private TaskList taskList;
    private String[] taskName;
    private String[] subtaskName;
    private int curTaskId = 0;
    private int curSubtaskId = 0;

    private boolean isVideo;

    private CheckBox cameraSwitch;

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

        /*
        taskList = TaskList.parseFromFile(getResources().openRawResource(R.raw.tasklist));
        TaskList.saveToLocalFile(taskList);
        taskList = TaskList.parseFromLocalFile();
         */
        // taskList = NetworkUtils.getTaskList(this, );

        loadTaskListViaNetwork();

        vibrator = (Vibrator) mContext.getSystemService(Context.VIBRATOR_SERVICE);

        recorder = new Recorder(this, new Recorder.RecorderListener() {
            @Override
            public void onTick(int tickCount, int times) {
                Log.e("TEST", "onTick " + tickCount);
                counter.setText(tickCount + " / " + times);
                vibrator.vibrate(VibrationEffect.createOneShot(200, 128));
            }

            @Override
            public void onFinish() {
                Log.e("TEST", "onFinish ");
                vibrator.vibrate(VibrationEffect.createOneShot(600, 128));
                enableButtons(false);
            }
        });
    }

    private void loadTaskListViaNetwork() {
        NetworkUtils.getAllTaskList(this, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                StringListBean taskLists = new Gson().fromJson(response.body(), StringListBean.class);
                if (taskLists.getResult().size() > 0) {
                    String taskListId = taskLists.getResult().get(0);
                    NetworkUtils.getTaskList(mContext, taskListId, 0, new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            taskList = new Gson().fromJson(response.body(), TaskList.class);
                            initView();
                        }
                    });
                }
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadTaskListViaNetwork();
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
        user.setText("a");
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

        if (taskName.length == 0) {
            subtaskName = new String[0];
        }
        else {
            subtaskName = taskList.getTask().get(curTaskId).getSubtaskName();
        }
        subtaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, subtaskName);
        subtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        subtaskSpinner.setAdapter(subtaskAdapter);
        subtaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                curSubtaskId = position;
                description.setText(subtaskName[curSubtaskId]);
                isVideo = taskList.getTask().get(curTaskId).getSubtask().get(curSubtaskId).isVideo() |
                          taskList.getTask().get(curTaskId).isAudio();
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
        configButton = findViewById(R.id.configButton);
        trainButton = findViewById(R.id.trainButton);

        startButton.setOnClickListener(view -> {
            enableButtons(true);
            recorder.start(
                    user.getText().toString(),
                    curTaskId,
                    curSubtaskId,
                    taskList
            );
        });

        stopButton.setOnClickListener(view -> {
            recorder.stop();
            enableButtons(false);
        });

        configButton.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, ConfigTaskActivity.class);
            startActivity(intent);
        });

        trainButton.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, TrainActivity.class);
            startActivity(intent);
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