package com.hcifuture.contextactionlibrary.collect.data;

import android.os.Build;

import androidx.annotation.RequiresApi;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class InformationContextData extends Data {
    private List<InformationData> data;

    public InformationContextData() {
        data = new ArrayList<>();
    }

    public void setData(List<InformationData> data) {
        this.data = data;
    }

    public List<InformationData> getData() {
        return data;
    }

    public void clear() {
        this.data.clear();
    }

    public void insert(InformationData single) {
        data.add(single);
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public InformationContextData deepClone() {
        InformationContextData data = new InformationContextData();
        getData().forEach(data::insert);
        return data;
    }
}
