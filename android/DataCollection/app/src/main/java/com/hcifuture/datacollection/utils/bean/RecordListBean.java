package com.hcifuture.datacollection.utils.bean;

import java.util.List;

public class RecordListBean {
    private List<RecordBean> recordList;

    public List<RecordBean> getRecordList() {
        return recordList;
    }

    public void setRecordList(List<RecordBean> recordList) {
        this.recordList = recordList;
    }

    public static class RecordBean {
        private String taskListId;
        private String taskId;
        private String subtaskId;
        private String recordId;
        private long timestamp;

        public String getTaskListId() {
            return taskListId;
        }

        public String getRecordId() {
            return recordId;
        }

        public String getSubtaskId() {
            return subtaskId;
        }

        public String getTaskId() {
            return taskId;
        }

        public long getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(long timestamp) {
            this.timestamp = timestamp;
        }

        public void setTaskListId(String taskListId) {
            this.taskListId = taskListId;
        }

        public void setRecordId(String recordId) {
            this.recordId = recordId;
        }

        public void setSubtaskId(String subtaskId) {
            this.subtaskId = subtaskId;
        }

        public void setTaskId(String taskId) {
            this.taskId = taskId;
        }
    }
}
