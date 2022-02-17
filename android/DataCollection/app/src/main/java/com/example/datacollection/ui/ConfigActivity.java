package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.widget.ListView;

import com.example.datacollection.R;
import com.example.datacollection.TaskList;
import com.example.datacollection.ui.adapter.TaskAdapter;

public class ConfigActivity extends AppCompatActivity {

    private ListView taskListView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_config);

        taskListView = findViewById(R.id.taskListView);
        TaskList taskList = TaskList.parseFromFile(getResources().openRawResource(R.raw.tasklist));

        TaskAdapter taskAdapter = new TaskAdapter(this, taskList);
        taskListView.setAdapter(taskAdapter);
    }
}