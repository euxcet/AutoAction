package com.example.contextactionlibrary.contextaction.context.informational;

import com.example.ncnnlibrary.communicate.event.BroadcastEvent;

import java.util.List;

public class Task {
    private int id;
    private String name;
    private String describe;
    private String mainPackage;
    private List<Page> pageList;
    private List<BroadcastEvent> actionSequence;
    private double threshold = 0.5;

    public Task(int id,String name,String describe,List<Page> pageList,List<BroadcastEvent> actionSequence) {
        this.id = id;
        this.name = name;
        this.describe = describe;
        this.pageList = pageList;
        this.actionSequence = actionSequence;
    }


    public boolean match(List<Page> pages,List<BroadcastEvent> actions) {
        int pageDistance = minDistancePage(pages, pageList);
        int actionDistance = minDistanceAction(actions, actionSequence);
        double pageRes = (2 * pageList.size() - pageDistance) / pageList.size() * 2;
        double actionRes = (2 * actionSequence.size() - actionDistance) / actionSequence.size() * 2;
        return (pageRes + actionRes) / 2 > threshold;
    }

    public static int minDistancePage(List<Page> l1, List<Page> l2) {
        int m = l1.size();
        int n = l2.size();
        int[][] dp = new int[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }

        for (int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                int c = 2;
                if (l1.get(i - 1).getId() == l2.get(j - 1).getId()) {
                    c = 0;
                }
                else if (l1.get(i).getPackageName().equals(l2.get(j).getPackageName())) {
                    c = 1;
                }
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + c);
            }
        }
        return dp[m][n];
    }

    public static int minDistanceAction(List<BroadcastEvent> l1, List<BroadcastEvent> l2) {
        int m = l1.size();
        int n = l2.size();
        int[][] dp = new int[m + 1][n + 1];
        for(int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        for(int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }

        for(int i = 1; i <= m; i++) {
            for(int j = 1; j <= n; j++) {
                int c = 2;
                if (l1.get(i - 1).getTag().equals(l2.get(j - 1).getTag())) {
                    c = 0;
                }
                else if (l1.get(i).getType().equals(l2.get(j).getType())) {
                    c = 1;
                }
                dp[i][j] = Math.min(Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1), dp[i - 1][j - 1] + c);
            }
        }
        return dp[m][n];
    }

    public String getDescribe() {
        return describe;
    }
}
