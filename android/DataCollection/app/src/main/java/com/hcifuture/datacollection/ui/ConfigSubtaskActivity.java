package com.hcifuture.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.hcifuture.datacollection.R;
import com.hcifuture.datacollection.utils.GlobalVariable;
import com.hcifuture.datacollection.utils.bean.TaskListBean;
import com.hcifuture.datacollection.ui.adapter.SubtaskAdapter;
import com.hcifuture.datacollection.utils.NetworkUtils;
import com.google.gson.Gson;
import com.lzy.okgo.callback.StringCallback;
import com.lzy.okgo.model.Response;

public class ConfigSubtaskActivity extends AppCompatActivity {
    private Context mContext;
    private ListView subtaskListView;
    private TaskListBean taskList;
    private SubtaskAdapter subtaskAdapter;
    private TextView taskNameView;
    private int task_id;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_subtask);
        mContext = this;

        Button backButton = findViewById(R.id.subtaskBackButton);
        backButton.setOnClickListener((v) -> this.finish());

        subtaskListView = findViewById(R.id.subtaskListView);

        Bundle bundle = getIntent().getExtras();
        task_id = bundle.getInt("task_id");

        Button addButton = findViewById(R.id.addSubtaskButton);
        addButton.setOnClickListener((v) -> {
            Bundle addBundle = new Bundle();
            addBundle.putInt("task_id", task_id);
            Intent intent = new Intent(ConfigSubtaskActivity.this, AddSubtaskActivity.class);
            intent.putExtras(addBundle);
            startActivity(intent);
        });

        Button configButton = findViewById(R.id.configTaskButton);
        configButton.setOnClickListener((v) -> {
            Bundle configBundle = new Bundle();
            configBundle.putInt("task_id", task_id);
            Intent intent = new Intent(ConfigSubtaskActivity.this, ModifyTaskActivity.class);
            intent.putExtras(configBundle);
            startActivity(intent);
        });

        taskNameView = findViewById(R.id.taskNameView);
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
                taskNameView.setText("Task name: " + taskList.getTask().get(task_id).getName());
                subtaskAdapter = new SubtaskAdapter(mContext, taskList, task_id);
                subtaskListView.setAdapter(subtaskAdapter);
            }
        });
    }
}