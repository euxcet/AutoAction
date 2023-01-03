package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import android.content.Context;
import android.graphics.Rect;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.hcifuture.contextactionlibrary.contextaction.ContextActionContainer;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

public class PageController {
    public static Map<String,List<Page>> pages = new HashMap<>();
    private static Map<Integer,Page> idToPage = new HashMap<>();
    private static HashSet<String> allFunctionWords = new HashSet<>();

    @RequiresApi(api = Build.VERSION_CODES.N)
    public static void initPages(Context context)
    {
        try
        {
            loadFunctionWords(context);
            InputStream inStream = new FileInputStream(ContextActionContainer.getSavePath() + "pages.csv");
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
    public static void loadFunctionWords(Context context)
    {
        try
        {
            InputStream inStream = new FileInputStream(ContextActionContainer.getSavePath() + "words.csv");
            BufferedReader br = new BufferedReader(new InputStreamReader(inStream));
            String line;
            while ((line = br.readLine()) != null) {
                String word = line.split("\t")[0];
                allFunctionWords.add(word);
            }
            br.close();
            inStream.close();

        }catch(Exception e)
        {
            e.printStackTrace();
        }
    }
    @RequiresApi(api = Build.VERSION_CODES.N)
    synchronized static void loadPages(StringBuilder sb)
    {
        allFunctionWords.clear();
        pages.clear();
        idToPage.clear();
        String[] lines = sb.toString().split("\n");
        for(String line : lines)
        {
            loadLine(line);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public static void loadLine(String line)
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

    synchronized static void loadWords(StringBuilder sb)
    {
        allFunctionWords.clear();
        String[] lines = sb.toString().split("\n");
        for(String line : lines)
        {
            String[] res = line.split("\t");
            allFunctionWords.add(res[0]);
        }
    }


    public static Page recognizePage(List<AccessibilityNodeInfoRecordFromFile> roots,String packageName)
    {
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

    public static Page recognizePage(HashSet<String> words,String packageName)
    {
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

    public static HashSet<String> getAllFunctionWords(List<AccessibilityNodeInfoRecordFromFile> roots)
    {
        HashSet<String> res = new HashSet<>();
        for(AccessibilityNodeInfoRecordFromFile root:roots)
        {
            getNodeFunctionWords(root,res);
            break;
        }
        Log.e("getAllFunctionWords",res.toString());
        return res;
    }

    public static void getNodeFunctionWords(AccessibilityNodeInfoRecordFromFile node, HashSet<String> set)
    {
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

    public static boolean judgeBound(AccessibilityNodeInfoRecordFromFile node)
    {
        if(!node._isVisibleToUser)
            return false;

        Rect r = new Rect();
        node.getBoundsInScreen(r);
        return !r.isEmpty();
    }

    public static Map<Integer, Page> getIdToPage() {
        return idToPage;
    }
}

