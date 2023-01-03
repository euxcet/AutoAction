package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import java.util.HashSet;
import java.util.Set;

public class Page {

    int id;
    String packageName;
    String title;


    public String getTitle() {
        return title;
    }

    HashSet<String> functionWords = new HashSet<>();

    double threshold = 0.8;

    public int getId() {
        return id;
    }

    public Page(int id, String packageName, String title, HashSet<String> functionWords)
    {
        this.id = id;
        this.packageName = packageName;
        this.title = title;
        this.functionWords = functionWords;
    }


    public double match(String packageName, Set<String> words)
    {
        if(!packageName.equals(this.packageName))
            return 0;

        HashSet<String> result = new HashSet<>();
        result.addAll(functionWords);
        result.retainAll(words);
        double res = ((double) result.size()/functionWords.size()+(double) result.size()/words.size())/2;
        return res;
    }
}
