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

    private TaskListBean mTaskList;

    private EditText mEditTextName;
    private EditText mEditTextTimes;
    private EditText mEditTextDuration;
    private CheckBox mCheckboxVideo;
    private CheckBox mCheckboxAudio;

    private int mTaskId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modify_task);
        this.mContext = this;
        this.mActivity = this;

        Bundle bundle = getIntent().getExtras();
        mTaskId = bundle.getInt("task_id");

        mEditTextName = findViewById(R.id.modify_task_edit_text_name);
        mEditTextTimes = findViewById(R.id.modify_task_edit_text_times);
        mEditTextDuration = findViewById(R.id.modify_task_edit_text_duration);
        mCheckboxVideo = findViewById(R.id.modify_task_video_switch);
        mCheckboxAudio = findViewById(R.id.modify_task_audio_switch);

        Button btnModify = findViewById(R.id.modify_task_btn_modify);
        Button btnCancel = findViewById(R.id.modify_task_btn_cancel);

        btnModify.setOnClickListener((v) -> modifyTask());
        btnCancel.setOnClickListener((v) -> this.finish());
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
                mTaskList = new Gson().fromJson(response.body(), TaskListBean.class);
                TaskListBean.Task task = mTaskList.getTasks().get(mTaskId);
                mEditTextName.setText(task.getName());
                mEditTextTimes.setText(String.valueOf(task.getTimes()));
                mEditTextDuration.setText(String.valueOf(task.getDuration()));
                mCheckboxVideo.setChecked(task.isVideo());
                mCheckboxAudio.setChecked(task.isAudio());
            }
        });
    }

    private void modifyTask() {
        TaskListBean.Task task = mTaskList.getTasks().get(mTaskId);
        task.setName(mEditTextName.getText().toString());
        task.setTimes(Integer.parseInt(mEditTextTimes.getText().toString()));
        task.setDuration(Integer.parseInt(mEditTextDuration.getText().toString()));
        task.setVideo(mCheckboxVideo.isChecked());
        task.setAudio(mCheckboxAudio.isChecked());

        NetworkUtils.updateTaskList(mContext, mTaskList, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mActivity.finish();
            }
        });
    }
}