package com.example.contextactionlibrary.contextaction.context.informational;

import java.util.HashSet;
import java.util.Set;

public class Page {
    private int id;
    private String packageName;
    private String title;
    private double threshold = 0.8;
    private HashSet<String> functionWords = new HashSet<>();

    public String getTitle() {
        return title;
    }

    public int getId() {
        return id;
    }

    public String getPackageName() {
        return packageName;
    }

    public Page(int id, String packageName, String title, HashSet<String> functionWords) {
        this.id = id;
        this.packageName = packageName;
        this.title = title;
        this.functionWords = functionWords;
    }


    public boolean match(String packageName, Set<String> words) {
        if(!packageName.equals(this.packageName))
            return false;

        HashSet<String> result = new HashSet<>();
        result.addAll(functionWords);
        result.retainAll(words);
        int res = result.size()/(functionWords.size()+words.size()-result.size());
        return res>0.8;
    }
}
