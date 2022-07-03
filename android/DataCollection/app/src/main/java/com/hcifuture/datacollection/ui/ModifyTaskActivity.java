package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Bundle;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

/**
 * The activity used to modify task settings.
 * Jumped from ConfigSubtaskActivity.
 */
public class ModifyTaskActivity extends AppCompatActivity {
    private Context mContext;
    private AppCompatActivity mActivity;

    private TaskListBean taskList;

    private EditText nameEditText;
    private EditText timesEditText;
    private EditText durationEditText;
    private CheckBox videoCheckbox;
    private CheckBox audioCheckbox;

    private int task_id;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modify_task);
        this.mContext = this;
        this.mActivity = this;

        Bundle bundle = getIntent().getExtras();
        task_id = bundle.getInt("task_id");

        nameEditText = findViewById(R.id.addTaskNameEdit);
        timesEditText = findViewById(R.id.addTaskTimesEdit);
        durationEditText = findViewById(R.id.addTaskDurationEdit);
        videoCheckbox = findViewById(R.id.addTaskVideoCheckbox);
        audioCheckbox = findViewById(R.id.addTaskAudioCheckbox);

        Button confirmButton = findViewById(R.id.addTaskConfirmButton);
        Button cancelButton = findViewById(R.id.addTaskCancelButton);

        cancelButton.setOnClickListener((v) -> this.finish());
        confirmButton.setOnClickListener((v) -> modifyTask());
    }

    @Override
    protected void onResume() {
        super.onResume();
        loadTaskListViaNetwork();
    }

    private void loadTaskListViaNetwork() {
        NetworkUtils.getTaskList(mContext, GlobalVariable.getInstance().getString("taskListId"), 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                taskList = new Gson().fromJson(response.body(), TaskListBean.class);
                TaskListBean.Task task = taskList.getTask().get(task_id);
                nameEditText.setText(task.getName());
                timesEditText.setText(String.valueOf(task.getTimes()));
                durationEditText.setText(String.valueOf(task.getDuration()));
                videoCheckbox.setChecked(task.isVideo());
                audioCheckbox.setChecked(task.isAudio());
            }
        });
    }

    private void modifyTask() {
        TaskListBean.Task task = taskList.getTask().get(task_id);
        task.setName(nameEditText.getText().toString());
        task.setTimes(Integer.parseInt(timesEditText.getText().toString()));
        task.setDuration(Integer.parseInt(durationEditText.getText().toString()));
        task.setVideo(videoCheckbox.isChecked());
        task.setAudio(audioCheckbox.isChecked());

        NetworkUtils.updateTaskList(mContext, taskList, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mActivity.finish();
            }
        });
    }
}