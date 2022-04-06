package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import android.content.Context;
import android.graphics.Rect;
import android.os.Build;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.BuildConfig;
import com.hcifuture.contextactionlibrary.R;

import java.io.BufferedReader;
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
    private static Map<String,List<Page>> pages = new HashMap<>();
    private static Map<Integer,Page> idToPage = new HashMap<>();
    private static HashSet<String> allFunctionWords = new HashSet<>();

    public static void initPages(Context context)
    {
        try
        {
            InputStream inStream = new FileInputStream(BuildConfig.SAVE_PATH + "pages.csv");
            BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
            String line;
            while ((line = br.readLine()) != null) {
                loadLine(line);
            }
            br.close();
            inStream.close();
        }catch(Exception e)
        {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private static void loadLine(String line)
    {
        try
        {
            if (line.startsWith("id"))
                return;
            String obj[] = line.split(",");
            if (obj.length < 4)
                return;

            String id = obj[0];
            String title = obj[1];
            String pkg = obj[2];
            String words = obj[3];
            if (obj.length > 4)
                words = line.substring(id.length() + title.length() + pkg.length() + 4,line.length()-1);
            String ws[] = words.split("##");
            HashSet<String> functionWordsSet = new HashSet<>();
            for (String word : ws) {
                functionWordsSet.add(word);
                allFunctionWords.add(word);
            }
            Integer id_ =Integer.parseInt(id);
            Page page = new Page(id_, pkg, title, functionWordsSet);
            List<Page> list = pages.getOrDefault(pkg, new ArrayList<>());
            list.add(page);
            pages.put(pkg, list);
            idToPage.put(id_,page);
        }
        catch(Exception e)
        {
            System.out.println(line);
            e.printStackTrace();
        }
    }

    public static Page recognizePage(List<AccessibilityNodeInfoRecordFromFile> roots,String packageName) {
        HashSet<String> words = getAllFunctionWords(roots);
        double max_sim = 0.5;
        Page res=null;
        if(pages.containsKey(packageName)) {
            for (Page page : pages.get(packageName)) {
                double sim = page.match(packageName, words);
                if (sim>max_sim) {
                    max_sim=sim;
                    res = page;
                }else if(sim==max_sim&&res!=null)
                {
                    if(res.functionWords.size()<page.functionWords.size())
                        res = page;
                }
            }
        }
        return res;
    }

    private static HashSet<String> getAllFunctionWords(List<AccessibilityNodeInfoRecordFromFile> roots) {
        HashSet<String> res = new HashSet<>();
        for(AccessibilityNodeInfoRecordFromFile root:roots)
        {
            getNodeFunctionWords(root,res);
        }
        return res;
    }

    private static void getNodeFunctionWords(AccessibilityNodeInfoRecordFromFile node, HashSet<String> set) {
        if(!judgeBound(node))
            return;
        if(node._isScrollable)
            set.add("SCROLLABLE");
        else if(node._className!=null && node._className.toString().contains("EditText"))
            set.add("EDIT_TEXT");
        else if(node._text!=null && !node._text.toString().equals(""))
        {
            String[] texts = node._text.toString().split("\n");
            for(String text:texts) {
                if (allFunctionWords.contains(text))
                    set.add(text);
            }
        }else if(node._contentDescription!=null && !node._contentDescription.toString().equals(""))
        {
            String[] texts = node._contentDescription.toString().split("\n");
            for(String text:texts) {
                if (allFunctionWords.contains(text))
                    set.add(text);
            }
        }
        for(AccessibilityNodeInfoRecordFromFile n: node.children)
            getNodeFunctionWords(n,set);
    }

    private static boolean judgeBound(AccessibilityNodeInfoRecordFromFile node)
    {
        if(!node._isVisibleToUser)
            return false;

        Rect r = new Rect();
        node.getBoundsInScreen(r);
        return !r.isEmpty();
    }

    public static Map<String, List<Page>> getPages() {
        return pages;
    }

    public static Map<Integer, Page> getIdToPage() {
        return idToPage;
    }
    public static HashSet<String> getAllFunctionWords() {
        return allFunctionWords;
    }
}
