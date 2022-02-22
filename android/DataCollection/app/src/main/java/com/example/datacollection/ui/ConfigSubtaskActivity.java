package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ListView;
import android.widget.TextView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.adapter.SubtaskAdapter;
import com.example.datacollection.ui.adapter.TaskAdapter;

public class ConfigSubtaskActivity extends AppCompatActivity {
    private ListView subtaskListView;
    private Button addButton;
    private Button backButton;
    private TaskList taskList;
    private SubtaskAdapter subtaskAdapter;
    private int task_id;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config_subtask);

        backButton = findViewById(R.id.subtaskBackButton);
        backButton.setOnClickListener((v) -> this.finish());

        subtaskListView = findViewById(R.id.subtaskListView);
        taskList = TaskList.parseFromLocalFile();

        Bundle bundle = getIntent().getExtras();
        task_id = bundle.getInt("task_id");

        addButton = findViewById(R.id.addSubtaskButton);
        addButton.setOnClickListener((v) -> {
            Bundle addBundle = new Bundle();
            addBundle.putInt("task_id", task_id);
            Intent intent = new Intent(ConfigSubtaskActivity.this, AddSubtaskActivity.class);
            intent.putExtras(addBundle);
            startActivity(intent);
        });

        TextView taskNameView = findViewById(R.id.taskNameView);
        if (taskList != null) {
            taskNameView.setText("Task name: " + taskList.getTask().get(task_id).getName());
        }

        subtaskAdapter = new SubtaskAdapter(this, taskList, task_id);
        subtaskListView.setAdapter(subtaskAdapter);
    }

    @Override
    protected void onResume() {
        super.onResume();
        taskList = TaskList.parseFromLocalFile();
        subtaskAdapter = new SubtaskAdapter(this, taskList, task_id);
        subtaskListView.setAdapter(subtaskAdapter);
    }
}