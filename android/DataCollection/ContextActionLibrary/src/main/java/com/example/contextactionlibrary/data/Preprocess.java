package com.example.contextactionlibrary.data;

public class Preprocess {

    private static Preprocess instance;

    public static Preprocess getInstance() {
        if (instance == null)
            instance = new Preprocess();
        return instance;
    }

    public Preprocess() {

    }
}
