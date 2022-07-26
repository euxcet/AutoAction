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

    private TaskListBean mTaskList;

    private EditText mEditTextName;
    private EditText mEditTextTimes;
    private EditText mEditTextDuration;
    private CheckBox mCheckboxVideo;
    private CheckBox mCheckboxAudio;
    private CheckBox mCheckboxFacing;

    private int mTaskId;
    private int mSubtaskId;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_modify_subtask);
        this.mContext = this;
        this.mActivity = this;

        Bundle bundle = getIntent().getExtras();
        mTaskId = bundle.getInt("task_id");
        mSubtaskId = bundle.getInt("subtask_id");

        mEditTextName = findViewById(R.id.modify_subtask_edit_text_name);
        mEditTextTimes = findViewById(R.id.modify_subtask_edit_text_times);
        mEditTextDuration = findViewById(R.id.modify_subtask_edit_text_duration);
        mCheckboxVideo = findViewById(R.id.modify_subtask_video_switch);
        mCheckboxAudio = findViewById(R.id.modify_subtask_audio_switch);
        mCheckboxFacing = findViewById(R.id.modify_subtask_video_facing);

        Button btnModify = findViewById(R.id.modify_subtask_btn_modify);
        Button btnCancel = findViewById(R.id.modify_subtask_btn_cancel);

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
                TaskListBean.Task.Subtask subtask = mTaskList.getTasks().get(mTaskId).getSubtasks().get(mSubtaskId);
                mEditTextName.setText(subtask.getName());
                mEditTextTimes.setText(String.valueOf(subtask.getTimes()));
                mEditTextDuration.setText(String.valueOf(subtask.getDuration()));
                mCheckboxVideo.setChecked(subtask.isVideo());
                mCheckboxAudio.setChecked(subtask.isAudio());
                mCheckboxFacing.setChecked(subtask.getLensFacing() == 0);
            }
        });
    }

    private void modifyTask() {
        TaskListBean.Task.Subtask subtask = mTaskList.getTasks().get(mTaskId).getSubtasks().get(mSubtaskId);
        subtask.setName(mEditTextName.getText().toString());
        subtask.setTimes(Integer.parseInt(mEditTextTimes.getText().toString()));
        subtask.setDuration(Integer.parseInt(mEditTextDuration.getText().toString()));
        subtask.setVideo(mCheckboxVideo.isChecked());
        subtask.setAudio(mCheckboxAudio.isChecked());
        subtask.setLensFacing(mCheckboxFacing.isChecked() ? 0 : 1);

        NetworkUtils.updateTaskList(mContext, mTaskList, 0, new StringCallback() {
            @Override
            public void onSuccess(Response<String> response) {
                mActivity.finish();
            }
        });
    }
}