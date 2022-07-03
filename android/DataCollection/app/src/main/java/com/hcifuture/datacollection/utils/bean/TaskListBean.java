package com.hcifuture.datacollection.utils.bean;

import com.hcifuture.datacollection.BuildConfig;
import com.hcifuture.datacollection.utils.FileUtils;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.io.StringWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

/**
 * Stores the meta data of a task list.
 * TaskList -> List(Task) -> List(List(Subtask))
 * CAUTION: this file would be converted to json, be careful for modifying variable names!
 */
public class TaskListBean implements Serializable {
    private String id;
    private String date;
    private String description;
    private List<Task> tasks;

    public enum FILE_TYPE {
        SENSOR,
        TIMESTAMP,
        AUDIO,
        VIDEO,
        SENSOR_BIN
    }

    @Deprecated
    public static TaskListBean parseFromFile(InputStream is) {
        try {
            Writer writer = new StringWriter();
            char[] buffer = new char[1024];
            try {
                BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
                int n;
                while ((n = reader.read(buffer)) != -1) {
                    writer.write(buffer, 0, n);
                }
            } finally {
                is.close();
            }

            String jsonString = writer.toString();
            Gson gson = new GsonBuilder().create();
            TaskListBean taskList = gson.fromJson(jsonString, TaskListBean.class);
            // taskList.updateSubtask();
            return taskList;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    @Deprecated
    public static TaskListBean parseFromLocalFile() {
        TaskListBean taskList = null;
        try {
            taskList = TaskListBean.parseFromFile(new FileInputStream(BuildConfig.SAVE_PATH + "tasklist.json"));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return taskList;
    }

    @Deprecated
    public static void saveToLocalFile(TaskListBean taskList) {
        FileUtils.writeStringToFile(new Gson().toJson(taskList), new File(BuildConfig.SAVE_PATH + "tasklist.json"));
    }

    public String[] getTaskNames() {
        List<Task> tasks = getTasks();
        int size = tasks.size();
        String[] taskNames = new String[size];
        for(int i = 0; i < size; i++) {
            Task t = tasks.get(i);
            taskNames[i] = t.getId() + ". " + t.getName();
        }
        return taskNames;
    }

    public String getTaskNameById(String taskId) {
        List<Task> tasks = getTasks();
        for(int i = 0; i < tasks.size(); i++) {
            Task t = tasks.get(i);
            if (t.getId().equals(taskId)) {
                return t.getName();
            }
        }
        return null;
    }

    public Task getTaskById(String taskId) {
        List<Task> tasks = getTasks();
        for(int i = 0; i < tasks.size(); i++) {
            Task t = tasks.get(i);
            if (t.getId().equals(taskId)) {
                return t;
            }
        }
        return null;
    }

    public void addTask(Task newTask) {
        tasks.add(newTask);
    }

    public static class Task implements Serializable {
        private String id;
        private String name;
        private int times;      // should be deleted
        private int duration;   // should be deleted
        private boolean audio;  // should be deleted
        public boolean video;   // should be deleted
        public List<Subtask> subtasks;

        public Task(String id, String name, int times, int duration, boolean audio, boolean video) {
            this.id = id;
            this.name = name;
            this.times = times;
            this.duration = duration;
            this.audio = audio;
            this.video = video;
            this.subtasks = new ArrayList<>();
        }

        public void addSubtask(Subtask newSubtask) {
            subtasks.add(newSubtask);
        }

        public String[] getSubtaskNames() {
            int size = getSubtasks().size();
            String[] subtaskNames = new String[size];
            for(int i = 0; i < size; i++) {
                Subtask t = getSubtasks().get(i);
                subtaskNames[i] = t.getId() + ". " + t.getName();
            }
            return subtaskNames;
        }

        public String getSubtaskNameById(String subtaskId) {
            List<Subtask> subtasks = getSubtasks();
            for(int i = 0; i < subtasks.size(); i++) {
                Subtask t = subtasks.get(i);
                if (t.getId().equals(subtaskId)) {
                    return t.getName();
                }
            }
            return null;
        }

        public static class Subtask implements Serializable {
            private String name;
            private String id;
            private int times;
            private int duration;
            private boolean audio;
            private boolean video;

            public Subtask(String id, String name, int times, int duration, boolean audio, boolean video) {
                this.id = id;
                this.name = name;
                this.times = times;
                this.duration = duration;
                this.audio = audio;
                this.video = video;
            }

            public void setTimes(int times) {
                this.times = times;
            }

            public void setVideo(boolean video) {
                this.video = video;
            }

            public void setName(String name) {
                this.name = name;
            }

            public void setId(String id) {
                this.id = id;
            }

            public void setDuration(int duration) {
                this.duration = duration;
            }

            public void setAudio(boolean audio) {
                this.audio = audio;
            }

            public String getName() {
                return name;
            }

            public int getTimes() {
                return times;
            }

            public String getId() {
                return id;
            }

            public long getDuration() {
                return duration;
            }

            public boolean isVideo() {
                return video;
            }

            public boolean isAudio() {
                return audio;
            }
        }

        public boolean isAudio() {
            return audio;
        }

        public boolean isVideo() {
            return video;
        }

        public int getDuration() {
            return duration;
        }

        public String getId() {
            return id;
        }

        public int getTimes() {
            return times;
        }

        public List<Subtask> getSubtasks() {
            return subtasks;
        }

        public String getName() {
            return name;
        }

        public void setAudio(boolean audio) {
            this.audio = audio;
        }

        public void setDuration(int duration) {
            this.duration = duration;
        }

        public void setId(String id) {
            this.id = id;
        }

        public void setName(String name) {
            this.name = name;
        }

        public void setSubtasks(List<Subtask> subtasks) {
            this.subtasks = subtasks;
        }

        public void setTimes(int times) {
            this.times = times;
        }

        public void setVideo(boolean video) {
            this.video = video;
        }
    }

    public List<Task> getTasks() {
        return tasks;
    }

    public String getId() {
        return id;
    }

    public String getDate() {
        return date;
    }

    public String getDescription() {
        return description;
    }

    public void setId(String id) {
        this.id = id;
    }

    public void setTasks(List<Task> tasks) {
        this.tasks = tasks;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public void setDescription(String description) {
        this.description = description;
    }
}
