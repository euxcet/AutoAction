package com.example.contextactionlibrary.contextaction.context.informational;

import android.content.Context;

import com.example.contextactionlibrary.BuildConfig;
import com.example.contextactionlibrary.R;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class PageController {
    private Map<String, List<Page>> pages = new HashMap<>();
    private HashSet<String> allFunctionWords = new HashSet<>();

    public PageController(Context context) {
        loadWordsSet(context);
        HashSet<String> functionWordsSet = new HashSet<>();
        functionWordsSet.add("朋友圈");
        List<Page> wechat = new ArrayList<>();
        wechat.add(new Page(0,"com.tencent.mm","朋友圈",functionWordsSet));
        pages.put("com.tencent.mm", wechat);

    }

    synchronized private void loadWords(StringBuilder sb)
    {
        allFunctionWords.clear();
        String[] lines = sb.toString().split("\n");
        for(String line : lines)
        {
            String[] res = line.split("\t");
            allFunctionWords.add(res[0]);
        }
    }

    public boolean loadWordsSet(Context context) {
        allFunctionWords = new HashSet<>();
        try {
            InputStream inStream = new FileInputStream(BuildConfig.SAVE_PATH + "words.csv");
            // InputStream inStream = context.getResources().openRawResource(R.raw.words);
            BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
            String line;
            StringBuilder stringBuilder = new StringBuilder();
            while ((line = br.readLine()) != null) {
                stringBuilder.append(line);
                stringBuilder.append("\n");
            }
            loadWords(stringBuilder);

            br.close();
            inStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return allFunctionWords.size()>0;

    }

    public Page recognizePage(List<AccessibilityNodeInfoRecordFromFile> roots,String packageName) {
        HashSet<String> words = getAllFunctionWords(roots);
        if (pages.containsKey(packageName)) {
            for (Page page : pages.get(packageName)) {
                if (page.match(packageName, words))
                    return page;
            }
        }
        return null;
    }

    public HashSet<String> getAllFunctionWords(List<AccessibilityNodeInfoRecordFromFile> roots) {
        HashSet<String> res = new HashSet<>();
        for (AccessibilityNodeInfoRecordFromFile root:roots) {
            getNodeFunctionWords(root,res);
        }
        return res;
    }

    public void getNodeFunctionWords(AccessibilityNodeInfoRecordFromFile node, HashSet<String> set) {
        if (node._text != null) {
            String text = node._text.toString();
            if (allFunctionWords.contains(text)) {
                set.add(text);
            }
        }
        for (AccessibilityNodeInfoRecordFromFile n : node.children) {
            getNodeFunctionWords(n, set);
        }
    }

    public Map<String, List<Page>> getPages() {
        return pages;
    }
}
