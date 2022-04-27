package com.hcifuture.contextactionlibrary.contextaction.context.informational;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Task {
    int id;
    String name;
    String pkg;
    String mainPackage;
    List<Page> pageList;
    List<Action> actionSequence;
    double threshold = 0.5;

    public Task(int id,String name,String pkg,List<Page> pageList,List<Action> actionSequence)
    {
        this.id =id;
        this.name = name;
        this.pkg = pkg;
        this.pageList = pageList;
        this.actionSequence = actionSequence;
    }


    public boolean match(List<Page> pages,List<Action> actions)
    {
        int pageDistance = minDistancePage(pages,pageList);
        int actionDistance = minDistanceAction(actions,actionSequence);

        double pageRes = 1;
        if(!pageList.isEmpty())
            pageRes=(pageList.size()-pageDistance)/pageList.size();
        double actionRes =1;
        if(!actionSequence.isEmpty())
            actionRes= (actionSequence.size()-actionDistance)/actionSequence.size();
        return (pageRes+actionRes)/2>threshold;
    }

    public static int minDistancePage(List<Page> ll1,List<Page> ll2)
    {
        List<Page> rl1 = new ArrayList<>(ll1);
        List<Page> rl2 = new ArrayList<>(ll2);

        Collections.reverse(rl1);
        Collections.reverse(rl2);

        int m = rl1.size();
        int n = rl2.size();
        int dp[][] = new int[m+1][n+1];
        for(int i=0;i<=m;i++)
            dp[i][0] = i;

        for(int i=0;i<=n;i++)
            dp[0][i] = i;

        for(int i=1;i<=m;i++)
        {
            for(int j=1;j<=n;j++)
            {
                int c = 1;
                if(rl1.get(i-1).id==rl2.get(j-1).id)
                    c=0;
                dp[i][j] = Math.min(Math.min(dp[i-1][j],dp[i][j-1]+1),dp[i-1][j-1]+c);
            }
        }
        return dp[m][n];
    }

    public static int minDistanceAction(List<Action> ll1,List<Action> ll2)
    {

        List<Action> rl1 = new ArrayList<>(ll1);
        List<Action> rl2 = new ArrayList<>(ll2);

        Collections.reverse(rl1);
        Collections.reverse(rl2);

        int m = rl1.size();
        int n = rl2.size();
        int dp[][] = new int[m+1][n+1];
        for(int i=0;i<=m;i++)
            dp[i][0] = i;

        for(int i=0;i<=n;i++)
            dp[0][i] = i;

        for(int i=1;i<=m;i++)
        {
            for(int j=1;j<=n;j++)
            {
                int c = 1;
                if(rl1.get(i-1).text.equals(rl2.get(j-1).text)&&rl1.get(i-1).type.equals(rl2.get(j-1).type))
                    c=0;
                dp[i][j] = Math.min(Math.min(dp[i-1][j],dp[i][j-1]+1),dp[i-1][j-1]+c);
            }
        }
        return dp[m][n];
    }


    public String getName() {
        return name;
    }
}
