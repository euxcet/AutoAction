package com.example.contextactionlibrary.contextaction.context.informational;

import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.util.Pair;
import android.view.accessibility.AccessibilityNodeInfo;
import android.view.accessibility.AccessibilityWindowInfo;

import androidx.annotation.RequiresApi;

import com.example.ncnnlibrary.communicate.config.RequestConfig;
import com.example.ncnnlibrary.communicate.listener.RequestListener;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class AccessibilityNodeInfoRecordFromFile {

    public static List<AccessibilityNodeInfoRecordFromFile> roots_ = new ArrayList<>();
    public static Map<String, AccessibilityNodeInfoRecordFromFile> idToRecord_ = new HashMap<>();
    public static Map<Integer, String> nodeInfoHashtoId_ = new HashMap<>();

    int windowLayer;

    public static List<AccessibilityNodeInfo> getRootsInActiveWindow(RequestListener requestListener) {
        RequestConfig config = new RequestConfig();
        config.putValue("getWindows", 0);
        List<AccessibilityWindowInfo> windows = (List<AccessibilityWindowInfo>)requestListener.onRequest(config).getObject("getWindows");
        List<AccessibilityNodeInfo> nodes = new ArrayList<>();
        if (windows == null || windows.isEmpty()) {
            return nodes;
        }
        Collections.sort(windows, (accessibilityWindowInfo, t1) -> {
            Rect bound1 = new Rect();
            Rect bound2 = new Rect();
            accessibilityWindowInfo.getBoundsInScreen(bound1);
            t1.getBoundsInScreen(bound2);
            int are1 = (bound1.bottom - bound1.top) * (bound1.right - bound1.left);
            int are2 = (bound2.bottom - bound2.top) * (bound2.right - bound2.left);
            if (are1 > are2) {
                return -1;
            } else if (are1 < are2) {
                return 1;
            } else if (are1 == are2) {
                return 0;
            }
            if (accessibilityWindowInfo.getParent() != null && t1.getParent() == null) {
                return -1;
            } else if (accessibilityWindowInfo.getParent() == null && t1.getParent() != null) {
                return 1;
            }
            if (accessibilityWindowInfo.isActive()) {
                return -1;
            } else if (t1.isActive()) {
                return 1;
            } else {
                return 0;
            }
        });
        for (AccessibilityWindowInfo window : windows) {
            if (window.getRoot() != null)
                nodes.add(window.getRoot());
        }
        return nodes;
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public static List<AccessibilityNodeInfoRecordFromFile> buildAllTrees(RequestListener requestListener, String activityName)
    {
        List<AccessibilityNodeInfoRecordFromFile> roots = new ArrayList<>();
        idToRecord_.clear();
        nodeInfoHashtoId_.clear();
        for(AccessibilityNodeInfo nodeInfo: getRootsInActiveWindow(requestListener))
        {
            if(nodeInfo==null)
                continue;
            AccessibilityNodeInfoRecordFromFile record = new AccessibilityNodeInfoRecordFromFile(nodeInfo, null, 0,0);
            if(nodeInfo.getWindow()!=null) {
                record.windowTitle = nodeInfo.getWindow().getTitle();
                record.windowType = nodeInfo.getWindow().getType();
                record.windowLayer = nodeInfo.getWindow().getLayer();
            }
            record.activityName = activityName;
            Log.e("RESULT", record.activityName + " " + record.windowTitle + " " + record.windowType);
            roots.add(record);
        }
        roots_ =new ArrayList<>(roots);
        return roots;
    }



    public static void clearTree(List<AccessibilityNodeInfoRecordFromFile> roots){
        for(AccessibilityNodeInfoRecordFromFile node:roots)
            clearSubTree(node);
    }

    private static void clearSubTree(AccessibilityNodeInfoRecordFromFile record){
        if(record == null){
            return;
        }
        for(AccessibilityNodeInfoRecordFromFile child: record.children){
            clearSubTree(child);
        }
        if(record.nodeInfo != null) {
            record.nodeInfo.recycle();
            record.nodeInfo = null;
        }
    }

    public AccessibilityNodeInfo nodeInfo;
    public List<AccessibilityNodeInfoRecordFromFile> children;
    public AccessibilityNodeInfoRecordFromFile parent;
    public int index;

    public String allTexts;
    public String allContents;
    public CharSequence windowTitle;
    public int windowType;
    public String _absoluteId;
    public static int DEPTH_LIMIT = 100;

    @RequiresApi(api = Build.VERSION_CODES.N)
    AccessibilityNodeInfoRecordFromFile(AccessibilityNodeInfo nodeInfo, AccessibilityNodeInfoRecordFromFile parent, int index, int depth) {
        _isClickable = nodeInfo.isClickable();
        _isScrollable = nodeInfo.isScrollable();
        _isLongClickable = nodeInfo.isLongClickable();
        _isEditable = nodeInfo.isEditable();
        _isCheckable = nodeInfo.isCheckable();
        _isEnabled = nodeInfo.isEnabled();
        _isChecked = nodeInfo.isChecked();
        _isFocused = nodeInfo.isFocused();
        _isPassword = nodeInfo.isPassword();
        _isAccessibilityFocused = nodeInfo.isAccessibilityFocused();
        _text = nodeInfo.getText();
        _contentDescription = nodeInfo.getContentDescription();
        _className = nodeInfo.getClassName();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            _hintText = nodeInfo.getHintText();
        }
        _drawingOrder = nodeInfo.getDrawingOrder();

        Rect bound=new Rect();
        nodeInfo.getBoundsInScreen(bound);

        top = bound.top;
        left = bound.left;
        bottom = bound.bottom;
        right = bound.right;

        _isSelected = nodeInfo.isSelected();
        _packageName = nodeInfo.getPackageName();
        _viewIdResourceName = nodeInfo.getViewIdResourceName();

        _isVisibleToUser = nodeInfo.isVisibleToUser();
        _isFocusable = nodeInfo.isFocusable();
        _isDismissable = nodeInfo.isDismissable();

        this.nodeInfo = nodeInfo;
        this.children = new ArrayList<>();
        this.parent = parent;
        this.index = index;
        this._absoluteId = get_absoluteId();

        if(nodeInfo != null && depth<DEPTH_LIMIT) {
            for (int i = 0; i < nodeInfo.getChildCount(); ++i) {
                AccessibilityNodeInfo crtNode = nodeInfo.getChild(i);
                if (crtNode == null) {
                    continue;
                }
                children.add(new AccessibilityNodeInfoRecordFromFile(crtNode, this, i,depth+1));
            }
        }else if(depth>=DEPTH_LIMIT)
        {
            System.out.println("tree too depth");
        }
    }

    public String get_absoluteId(){
        if(getClassName()==null)
            return "";
        String res = "";
        if(parent == null){
            res = getClassName().toString();
        } else {
            res = parent._absoluteId + "|" + String.valueOf(index) + ";" + getClassName().toString();
        }
        AccessibilityNodeInfoRecordFromFile.idToRecord_.put(res, this);
        AccessibilityNodeInfoRecordFromFile.nodeInfoHashtoId_.put(this.nodeInfo.hashCode(),res);
        return res;
    }

    public AccessibilityNodeInfoRecordFromFile getParent(){
        return parent;
    }


    public int getChildCount(){
        return children.size();
    }


    public AccessibilityNodeInfoRecordFromFile getChild(int index){
        return children.get(index);
    }



    public boolean performAction(int action){
        return nodeInfo.performAction(action);
    }

    public boolean performAction(int action, Bundle info){
        return nodeInfo.performAction(action, info);
    }


    public AccessibilityWindowInfo getWindow(){
        return nodeInfo.getWindow();
    }


    public List<AccessibilityNodeInfoRecordFromFile> findAccessibilityNodeInfosByText(String str){
        List<AccessibilityNodeInfoRecordFromFile> res = new ArrayList<>();
        if(Objects.equals(getText().toString(), str)){
            res.add(this);
        }

        for(AccessibilityNodeInfoRecordFromFile child: children){
            res.addAll(child.findAccessibilityNodeInfosByText(str));
        }

        return res;
    }

    public AccessibilityNodeInfo getNodeInfo(){
        return nodeInfo;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }

        if(this.nodeInfo != null && ((AccessibilityNodeInfoRecordFromFile) obj).nodeInfo != null){
            return this.nodeInfo.equals(((AccessibilityNodeInfoRecordFromFile) obj).nodeInfo);
        } else if(this.nodeInfo == null && ((AccessibilityNodeInfoRecordFromFile) obj).nodeInfo == null){
            return this == obj;
        } else {
            return false;
        }

    }

    public int getIndex(){
        return index;
    }

    public List<AccessibilityNodeInfoRecordFromFile> getChildren() {
        return children;
    }


    public List<AccessibilityNodeInfoRecordFromFile> findNodeByViewIdResourceName(String str){
        List<AccessibilityNodeInfoRecordFromFile> res = new ArrayList<>();
        String crtId = getViewIdResourceName() == null? "": getViewIdResourceName().toString();
        if(Objects.equals(crtId, str)){
            res.add(this);
        }
        for(AccessibilityNodeInfoRecordFromFile child: children){
            res.addAll(child.findNodeByViewIdResourceName(str));
        }
        return res;

    }

    public boolean isMeaningful(){
        if(isCheckable() || isEditable() || isScrollable() || isLongClickable() || isClickable()){
            return true;
        }
        if(getText() != null && getText().length() > 0){
            return true;
        }
        if(getContentDescription() != null && getContentDescription().length() > 0){
            return true;
        }
        if(getViewIdResourceName() != null && getViewIdResourceName().length() > 0){
            return true;
        }
        if(children.size() != 1){
            return true;
        }
        return false;
    }

    public Pair<AccessibilityNodeInfoRecordFromFile, Integer> moveToMeaningfulChild(){
        AccessibilityNodeInfoRecordFromFile crtNode = this;
        int countSkipNum = 0;
        while (!crtNode.isMeaningful()){
            countSkipNum += 1;
            crtNode = crtNode.getChild(0);
        }
        return new Pair<>(crtNode, countSkipNum);
    }

    public String getAllTexts() {
        if (allTexts != null)
            return allTexts;
        allTexts = getText() == null? "": getText().toString();
        for(AccessibilityNodeInfoRecordFromFile child: children){

            allTexts += child.getAllTexts()==null?"":child.getAllTexts();
        }
        if(allTexts.isEmpty())
            return null;
        return allTexts;
    }

    public String getReadableText()
    {
        if(this.getText()!=null&&!this.getText().toString().isEmpty())
            return this.getText().toString();
        else if(this.getContentDescription()!=null&&!this.getContentDescription().toString().isEmpty())
            return this.getContentDescription().toString();
        else if(this.getAllTexts()!=null&&!this.getAllTexts().isEmpty())
            return this.getAllTexts();
        else
            return this.getAllContents();
    }

    public String getAllContents(){
        if (allContents != null)
            return allContents;
        allContents = getContentDescription() == null? "": getContentDescription().toString();
        for(AccessibilityNodeInfoRecordFromFile child: children){
            allContents += child.getAllContents()==null?"":child.getAllContents();
        }
        return allContents;
    }

    public AccessibilityNodeInfoRecordFromFile getNodeByRelativeId(String relativeId){
        String[] subIdList = relativeId.split(";");
        AccessibilityNodeInfoRecordFromFile crtNode = this;
        for(int i = 0; i < subIdList.length - 1; ++ i){
            String subId = subIdList[i];
            String[] subIdSplited = subId.split("\\|");
            if(!crtNode.getClassName().toString().equals(subIdSplited[0])){
                return null;
            }

            int intendedIndex = Integer.valueOf(subIdSplited[1]);
            AccessibilityNodeInfoRecordFromFile targetChild = null;
            for(AccessibilityNodeInfoRecordFromFile child: crtNode.children){
                if (child.index == intendedIndex){
                    targetChild = child;
                    break;
                }
            }

            if(targetChild == null){
                return null;
            }
            crtNode = targetChild;
        }
        if(!crtNode.getClassName().toString().equals(subIdList[subIdList.length - 1])){
            return null;
        }
        return crtNode;
    }

    // 从文件中载入 ui 树 用来对程序进行验证
    boolean _isClickable;
    boolean _isScrollable;
    boolean _isLongClickable;
    boolean _isEditable;
    boolean _isCheckable;
    boolean _isEnabled;
    boolean _isChecked;
    boolean _isFocused;
    boolean _isPassword;
    boolean _isAccessibilityFocused;
    boolean _isPrivate;

    CharSequence _text;  // nullable
    CharSequence _contentDescription;  // nullable
    CharSequence _className;
    CharSequence _hintText;

    int _drawingOrder;

    int top;
    int left;
    int bottom;
    int right;

    boolean _isSelected;
    CharSequence _packageName;
    CharSequence _viewIdResourceName;  // nullable

    boolean _isVisibleToUser;
    boolean _isFocusable;
    boolean _isDismissable;

    public String activityName;

    public boolean isChecked(){
        return _isChecked;
    }

    public boolean isFocused(){
        return _isFocused;
    }

    public boolean isPassword(){
        return _isPassword;
    }

    public boolean isAccessibilityFocused(){
        return _isAccessibilityFocused;
    }

    public boolean isClickable() {
        return _isClickable;
    }

    public boolean isScrollable() {
        return _isScrollable;
    }

    public boolean isLongClickable() {
        return _isLongClickable;
    }

    public boolean isEditable() {
        return _isEditable;
    }

    public boolean isEnabled(){
        return _isEnabled;
    }

    public boolean isCheckable() {
        return _isCheckable;
    }


    public CharSequence getText() {
        return _text;
    }

    public CharSequence getContentDescription() {
        return _contentDescription;
    }

    public CharSequence getClassName() {
        return _className;
    }

    public void getBoundsInScreen(Rect r) {
        r.left = left;
        r.right = right;
        r.top = top;
        r.bottom = bottom;
    }

    public boolean isSelected() {
        return _isSelected;
    }

    public CharSequence getPackageName() {
        return _packageName;
    }

    public CharSequence getViewIdResourceName() {
        return _viewIdResourceName;
    }

    public boolean isVisibleToUser() {
        return _isVisibleToUser;
    }

    public boolean isFocusable() {
        return _isFocusable;
    }

    public boolean isDismissable() {
        return _isDismissable;
    }

    public int getDrawingOrder() {
        return _drawingOrder;
    }

    public CharSequence getHintText() {
        return _hintText;
    }

    public boolean isPrivate(){return _isPrivate;}

}
