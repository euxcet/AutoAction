package com.example.datacollection.utils.bean;

import java.util.List;

public class TrainListBean {
    private List<TrainBean> trainList;

    public static class TrainBean {
        private String id;
        private String name;

        public String getId() {
            return id;
        }

        public String getName() {
            return name;
        }

        public void setId(String id) {
            this.id = id;
        }

        public void setName(String name) {
            this.name = name;
        }
    }

    public List<TrainBean> getTrainList() {
        return trainList;
    }

    public void setTrainList(List<TrainBean> trainList) {
        this.trainList = trainList;
    }
}
