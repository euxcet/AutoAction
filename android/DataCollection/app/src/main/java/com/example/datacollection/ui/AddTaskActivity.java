package com.example.datacollection.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Bundle;
import android.util.Log;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;

import com.example.datacollection.BuildConfig;
import com.example.datacollection.R;
import com.example.datacollection.TaskList;

import java.io.FileInputStream;

public class AddTaskActivity extends AppCompatActivity {
    private TaskList taskList;

    private EditText nameEditText;
    private EditText timesEditText;
    private EditText durationEditText;
    private CheckBox videoCheckbox;
    private CheckBox audioCheckbox;

    private Button confirmButton;
    private Button cancelButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_add_task);

        nameEditText = findViewById(R.id.addTaskNameEdit);
        timesEditText = findViewById(R.id.addTaskTimesEdit);
        durationEditText = findViewById(R.id.addTaskDurationEdit);
        videoCheckbox = findViewById(R.id.addTaskVideoCheckbox);
        audioCheckbox = findViewById(R.id.addTaskAudioCheckbox);

        confirmButton = findViewById(R.id.addTaskConfirmButton);

        cancelButton = findViewById(R.id.addTaskCancelButton);
        cancelButton.setOnClickListener((v) -> this.finish());
        confirmButton.setOnClickListener((v) -> {
            taskList = TaskList.parseFromLocalFile();
            TaskList.Task newTask = new TaskList.Task(
                    0,
                    nameEditText.getText().toString(),
                    Integer.parseInt(timesEditText.getText().toString()),
                    Integer.parseInt(durationEditText.getText().toString()),
                    videoCheckbox.isChecked(),
                    audioCheckbox.isChecked());
            taskList.addTask(newTask);
            taskList.resetId();
            TaskList.saveToLocalFile(taskList);
            this.finish();
        });
    }
}