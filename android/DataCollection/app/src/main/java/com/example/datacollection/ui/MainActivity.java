package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.example.datacollection.BuildConfig;
import com.example.datacollection.R;
import com.example.datacollection.contextaction.ContextActionLoader;
import com.example.datacollection.utils.FileUtils;
import com.example.datacollection.utils.bean.TaskListBean;
import com.example.datacollection.TransferData;
import com.example.datacollection.data.Recorder;
import com.example.datacollection.utils.NetworkUtils;
import com.example.datacollection.utils.bean.StringListBean;
import com.example.ncnnlibrary.communicate.BuiltInContextEnum;
import com.example.ncnnlibrary.communicate.SensorType;
import com.example.ncnnlibrary.communicate.config.ActionConfig;
import com.example.ncnnlibrary.communicate.config.ContextConfig;
import com.example.ncnnlibrary.communicate.config.RequestConfig;
import com.example.ncnnlibrary.communicate.listener.ActionListener;
import com.example.ncnnlibrary.communicate.BuiltInActionEnum;
import com.example.ncnnlibrary.communicate.listener.ContextListener;
import com.example.ncnnlibrary.communicate.listener.RequestListener;
import com.example.ncnnlibrary.communicate.result.RequestResult;
import com.google.gson.Gson;
import com.lzy.okgo.callback.FileCallback;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

import java.io.File;
import java.util.Arrays;

import dalvik.system.DexClassLoader;
import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.EasyPermissions;
import pub.devrel.easypermissions.PermissionRequest;

public class MainActivity extends AppCompatActivity {

    private Context mContext;
    private AppCompatActivity mActivity;
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
    private TaskListBean taskList;
    private String[] taskName;
    private String[] subtaskName;
    private int curTaskId = 0;
    private int curSubtaskId = 0;

    private boolean isVideo;

    private CheckBox cameraSwitch;

    private Recorder recorder;

    private DexClassLoader classLoader;

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

    private ContextActionLoader loader;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        requestPermissions();

        mContext = this;
        mActivity = this;

        transferData = TransferData.getInstance();
        vibrator = (Vibrator) getSystemService(Context.VIBRATOR_SERVICE);

        loadTaskListViaNetwork();

        vibrator = (Vibrator) mContext.getSystemService(Context.VIBRATOR_SERVICE);

        recorder = new Recorder(this, new Recorder.RecorderListener() {
            @Override
            public void onTick(int tickCount, int times) {
                counter.setText(tickCount + " / " + times);
                vibrator.vibrate(VibrationEffect.createOneShot(200, 128));
            }

            @Override
            public void onFinish() {
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
                            taskList = new Gson().fromJson(response.body(), TaskListBean.class);
                            initView();
                        }
                    });
                }
            }
        });
        loadContextActionLibrary();
    }

    private RequestResult handleRequest(RequestConfig config) {
        RequestResult result = new RequestResult();
        String packageName = config.getString("getAppTapBlockValue");
        if (packageName != null) {
            result.putValue("getAppTapBlockValueResult", 0);
        }

        Boolean request = config.getBoolean("getCanTapTap");
        if (request != null) {
            result.putValue("getCanTapTapResult", true);
        }

        request = config.getBoolean("getCanTopTap");
        if (request != null) {
            result.putValue("getCanTopTapResult", true);
        }
        return result;
    }

    private void loadContextActionLibrary() {
        NetworkUtils.downloadFile(this, "classes.dex", new FileCallback() {
            @Override
            public void onSuccess(Response<File> response) {
                File file = response.body();
                File saveFile = new File(BuildConfig.SAVE_PATH, "classes.dex");
                FileUtils.copy(file, saveFile);

                final File tmpDir = getDir("dex", 0);
                classLoader = new DexClassLoader(BuildConfig.SAVE_PATH + "classes.dex", tmpDir.getAbsolutePath(), null, this.getClass().getClassLoader());
                loader = new ContextActionLoader(mContext, classLoader);

                ActionConfig tapTapConfig = new ActionConfig();
                tapTapConfig.setAction(BuiltInActionEnum.TapTap);
                tapTapConfig.putValue("SeqLength", 50);
                tapTapConfig.setSensorType(Arrays.asList(SensorType.IMU));

                ActionListener actionListener = action ->
                        mActivity.runOnUiThread(
                                () -> Toast.makeText(mContext, action.getAction(), Toast.LENGTH_SHORT).show()
                        );

                ContextConfig proximityConfig = new ContextConfig();
                proximityConfig.setContext(BuiltInContextEnum.Proximity);
                proximityConfig.setSensorType(Arrays.asList(SensorType.PROXIMITY));

                ContextListener contextListener = context ->
                        mActivity.runOnUiThread(
                                () -> Toast.makeText(mContext, context.getContext(), Toast.LENGTH_SHORT).show()
                        );

                RequestListener requestListener = config -> handleRequest(config);
                loader.startDetection(Arrays.asList(tapTapConfig), actionListener, Arrays.asList(proximityConfig), contextListener);
            }
        });
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadTaskListViaNetwork();
        /*
        IntentFilter filter = new IntentFilter();
        filter.addAction("contextactionlibrary");
        mContext.registerReceiver(new BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                String action = intent.getExtras().getString("Action");
                if (action.equals("TapTap")) {
                    Toast.makeText(mContext, "TapTap", Toast.LENGTH_SHORT).show();
                }
            }
        }, filter);
         */
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

        /*
        NetworkUtils.downloadFile(mContext, "best.bin", new FileCallback() {
            @Override
            public void onSuccess(Response<File> response) {
                File file = response.body();
                File saveFile = new File(BuildConfig.SAVE_PATH, "best.bin");
                FileUtils.copy(file, saveFile);

                NetworkUtils.downloadFile(mContext, "best.param", new FileCallback() {
                    @Override
                    public void onSuccess(Response<File> response) {
                        File file = response.body();
                        File saveFile = new File(BuildConfig.SAVE_PATH, "best.param");
                        FileUtils.copy(file, saveFile);
                    }
                });
            }
        });
         */

        /*
        NcnnInstance.init(this,
                BuildConfig.SAVE_PATH + "best.param",
                BuildConfig.SAVE_PATH + "best.bin",
                4,
                128,
                6,
                1,
                2);
        NcnnInstance ncnnInstance = NcnnInstance.getInstance();
        ncnnInstance.print();
        float[] data = new float[128 * 6];
        Arrays.fill(data, 0.1f);
        Log.e("result", ncnnInstance.actionDetect(data) + " ");
        */

    }

    private void enableButtons(boolean isRecording) {
        startButton.setEnabled(!isRecording);
        stopButton.setEnabled(isRecording);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (loader != null) {
            loader.stopDetection();
        }
        if (vibrator != null) {
            vibrator.cancel();
        }
    }
}