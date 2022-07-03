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
 * The activity to modify the setting of a subtask.
 * Jumped from the subtask in SubtaskAdapter.
 */
public class ModifySubtaskActivity extends AppCompatActivity {
    private Context mContext;
    private AppCompatActivity mActivity;

    private TaskListBean taskList;

    private EditText nameEditText;
    private EditText timesEditText;
    private EditText durationEditText;
    private CheckBox videoCheckbox;
    private CheckBox audioCheckbox;

    private int task_id;
    private int subtask_id;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modify_subtask);
        this.mContext = this;
        this.mActivity = this;

        Bundle bundle = getIntent().getExtras();
        task_id = bundle.getInt("task_id");
        subtask_id = bundle.getInt("subtask_id");

        nameEditText = findViewById(R.id.addSubtaskNameEdit);
        timesEditText = findViewById(R.id.addSubtaskTimesEdit);
        durationEditText = findViewById(R.id.addSubtaskDurationEdit);
        videoCheckbox = findViewById(R.id.addSubtaskVideoCheckbox);
        audioCheckbox = findViewById(R.id.addSubtaskAudioCheckbox);

        Button confirmButton = findViewById(R.id.addSubtaskConfirmButton);
        Button cancelButton = findViewById(R.id.addSubtaskCancelButton);

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
                TaskListBean.Task.Subtask subtask = taskList.getTask().get(task_id).getSubtask().get(subtask_id);
                nameEditText.setText(subtask.getName());
                timesEditText.setText(String.valueOf(subtask.getTimes()));
                durationEditText.setText(String.valueOf(subtask.getDuration()));
                videoCheckbox.setChecked(subtask.isVideo());
                audioCheckbox.setChecked(subtask.isAudio());
            }
        });
    }

    private void modifyTask() {
        TaskListBean.Task.Subtask subtask = taskList.getTask().get(task_id).getSubtask().get(subtask_id);
        subtask.setName(nameEditText.getText().toString());
        subtask.setTimes(Integer.parseInt(timesEditText.getText().toString()));
        subtask.setDuration(Integer.parseInt(durationEditText.getText().toString()));
        subtask.setVideo(videoCheckbox.isChecked());
        subtask.setAudio(audioCheckbox.isChecked());

        NetworkUtils.updateTaskList(mContext, taskList, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mActivity.finish();
            }
        });
    }
}