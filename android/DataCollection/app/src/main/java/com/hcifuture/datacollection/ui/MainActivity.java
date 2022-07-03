package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.provider.Settings;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.inference.Inferencer;
import com.hcifuture.datacollection.service.MainService;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.data.Recorder;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.StringListBean;
import com.hcifuture.datacollection.visual.RecordListActivity;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.EasyPermissions;
import pub.devrel.easypermissions.PermissionRequest;

public class MainActivity extends AppCompatActivity {

    private Context mContext;
    private AppCompatActivity mActivity;
    private Vibrator mVibrator;

    // ui
    private EditText mUserText;
    private Button mBtnStart;
    private Button mBtnStop;
    private TextView mTaskDescription;
    private TextView mTaskCounter;

    private Spinner mTaskSpinner;
    private ArrayAdapter<String> mTaskAdapter;

    private Spinner mSubtaskSpinner;
    private ArrayAdapter<String> mSubtaskAdapter;

    // task
    private TaskListBean mTaskList;  // queried from the backend
    private String[] mTaskName;
    private String[] mSubtaskName;
    private int mCurrentTaskId = 0;
    private int mCurrentSubtaskId = 0;

    private boolean mIsVideo;

    private CheckBox mCameraSwitch;

    private Recorder mRecorder;

    // permission
    private static final int RC_PERMISSIONS = 0;
    private String[] mPermissions = new String[]{
            Manifest.permission.INTERNET,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.CAMERA,
            Manifest.permission.VIBRATE,
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.ACCESS_COARSE_LOCATION,
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACTIVITY_RECOGNITION
    };

    private Inferencer mInferencer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // ask for permissions
        requestPermissions();

        mContext = this;
        mActivity = this;

        // vibrate to indicate data collection progress
        mVibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);
        mVibrator = (Vibrator) mContext.getSystemService(Context.VIBRATOR_SERVICE);

        mRecorder = new Recorder(this, new Recorder.RecorderListener() {
            @Override
            public void onTick(int tickCount, int times) {
                mTaskCounter.setText(tickCount + " / " + times);
                mVibrator.vibrate(VibrationEffect.createOneShot(200, 128));
            }

            @Override
            public void onFinish() {
                mVibrator.vibrate(VibrationEffect.createOneShot(600, 128));
                enableButtons(false);
            }
        });

        // jump to accessibility settings
        Button accessibilityButton = findViewById(R.id.btn_access);
        accessibilityButton.setOnClickListener((v) -> {
            Intent settingIntent = new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS);
            startActivity(settingIntent);
        });

        Button upgradeButton = findViewById(R.id.btn_upgrade);
        upgradeButton.setOnClickListener((v) -> {
            MainService.getInstance().upgrade();
        });

        // goto test activity.
        Button testButton = findViewById(R.id.btn_test);
        testButton.setOnClickListener((v) -> {
            Intent intent = new Intent(MainActivity.this, TestModelActivity.class);
            startActivity(intent);
        });

        mInferencer = Inferencer.getInstance();
        mInferencer.start(this);
    }

    /**
     * Called in onResume().
     */
    private void loadTaskListViaNetwork() {
        NetworkUtils.getAllTaskList(this, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                StringListBean taskLists = new Gson().fromJson(response.body(), StringListBean.class);
                if (taskLists.getResult().size() > 0) {
                    String taskListId = taskLists.getResult().get(0);
                    GlobalVariable.getInstance().putString("taskListId", taskListId);
                    NetworkUtils.getTaskList(mContext, taskListId, 0, new StringCallback() {
                        @Override
                        public void onSuccess(Response<String> response) {
                            mTaskList = new Gson().fromJson(response.body(), TaskListBean.class);
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

    /**
     * Pop up dialog windows to ask users for system permissions.
     */
    @AfterPermissionGranted(RC_PERMISSIONS)
    private void requestPermissions() {
        if (EasyPermissions.hasPermissions(this, mPermissions)) {
            // have permissions
        } else {
            // no permissions, request dynamically
            EasyPermissions.requestPermissions(
                    new PermissionRequest.Builder(this, RC_PERMISSIONS, mPermissions)
                            .setRationale(R.string.rationale)
                            .setPositiveButtonText(R.string.rationale_ask_ok)
                            .setNegativeButtonText(R.string.rationale_ask_cancel)
                            .setTheme(R.style.Theme_AppCompat)
                            .build());
        }
    }

    /**
     * Init the status of all UI components in main activity.
     * Called in loadTaskListViaNetwork().
     */
    private void initView() {
        mUserText = findViewById(R.id.user_text);
        mUserText.setText(R.string.default_user);
        mTaskDescription = findViewById(R.id.task_description);
        mTaskCounter = findViewById(R.id.task_counter);

        // Spinner
        mTaskSpinner = findViewById(R.id.task_spinner);
        mSubtaskSpinner = findViewById(R.id.subtask_spinner);

        // choose tasks and subtasks
        mTaskName = mTaskList.getTaskName();
        mTaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, mTaskName);
        mTaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mTaskSpinner.setAdapter(mTaskAdapter);
        mTaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mCurrentTaskId = position;
                mSubtaskName = mTaskList.getTask().get(mCurrentTaskId).getSubtaskName();
                mSubtaskAdapter = new ArrayAdapter<>(mContext, android.R.layout.simple_spinner_item, mSubtaskName);
                mSubtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                mSubtaskSpinner.setAdapter(mSubtaskAdapter);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        if (mTaskName.length == 0) {
            mSubtaskName = new String[0];
        }
        else {
            mSubtaskName = mTaskList.getTask().get(mCurrentTaskId).getSubtaskName();
        }
        mSubtaskAdapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, mSubtaskName);
        mSubtaskAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        mSubtaskSpinner.setAdapter(mSubtaskAdapter);
        mSubtaskSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                mCurrentSubtaskId = position;
                mTaskDescription.setText(mSubtaskName[mCurrentSubtaskId]);
                mIsVideo = mTaskList.getTask().get(mCurrentTaskId).getSubtask()
                        .get(mCurrentSubtaskId).isVideo() |
                        mTaskList.getTask().get(mCurrentTaskId).isAudio();
                mCameraSwitch.setChecked(mIsVideo);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
            }
        });

        // whether to record the video
        mCameraSwitch = findViewById(R.id.video_switch);
        mCameraSwitch.setOnCheckedChangeListener((compoundButton, b) -> {
            mRecorder.setCamera(b);
        });
        mCameraSwitch.setEnabled(false); // disabled

        mBtnStart = findViewById(R.id.btn_start);
        mBtnStop = findViewById(R.id.btn_restart);

        Button configButton = findViewById(R.id.btn_config);
        Button trainButton = findViewById(R.id.btn_train);
        Button visualButton = findViewById(R.id.btn_visual);

        // click the start button to start recorder
        mBtnStart.setOnClickListener(view -> {
            enableButtons(true);
            mRecorder.start(
                    mUserText.getText().toString(),
                    mCurrentTaskId,
                    mCurrentSubtaskId,
                    mTaskList
            );
        });

        // click the stop button to end recording
        mBtnStop.setOnClickListener(view -> {
            mRecorder.interrupt();
            enableButtons(false);
        });

        // goto config task activity
        configButton.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, ConfigTaskActivity.class);
            startActivity(intent);
        });

        // goto train activity
        trainButton.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, TrainActivity.class);
            startActivity(intent);
        });

        // goto record list activity
        visualButton.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, RecordListActivity.class);
            startActivity(intent);
        });

        // set the default status of the start and end buttons
        enableButtons(false);
    }

    /**
     * Set the availability of the start and stop buttons.
     * Ensures the status of these two buttons are opposite.
     * @param isRecording Whether the current task is ongoing.
     */
    private void enableButtons(boolean isRecording) {
        mBtnStart.setEnabled(!isRecording);
        mBtnStop.setEnabled(isRecording);
    }

    /**
     * Cancel the vibrator.
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mVibrator != null) {
            mVibrator.cancel();
        }
    }
}