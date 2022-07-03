package com.hcifuture.datacollection.utils.bean;

import java.util.List;

/**
 * Stores the meta data of trains.
 */
public class TrainListBean {
    private List<TrainBean> trainList;

    public static class TrainBean {
        private String id;
        private String name;
        private String status;
        private String taskListId;
        private List<String> taskIdList;
        private long timestamp;

        public String getId() {
            return id;
        }

        public String getName() {
            return name;
        }

        public String getStatus() {
            return status;
        }

        public List<String> getTaskIdList() {
            return taskIdList;
        }

        public String getTaskListId() {
            return taskListId;
        }

        public long getTimestamp() {
            return timestamp;
        }

        public void setId(String id) {
            this.id = id;
        }

        public void setName(String name) {
            this.name = name;
        }

        public void setStatus(String status) {
            this.status = status;
        }

        public void setTaskIdList(List<String> taskIdList) {
            this.taskIdList = taskIdList;
        }

        public void setTaskListId(String taskListId) {
            this.taskListId = taskListId;
        }

        public void setTimestamp(long timestamp) {
            this.timestamp = timestamp;
        }
    }

    public List<TrainBean> getTrainList() {
        return trainList;
    }

    public void setTrainList(List<TrainBean> trainList) {
        this.trainList = trainList;
    }
}
