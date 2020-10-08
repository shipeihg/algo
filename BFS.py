
"""
BFS题目通常要你求一个 ”最小值“ 
一般是按照队列的顺序走，【中途】满足条件，则一定是最小值


广度优先搜索一层一层遍历，每一层得到的所有新节点，要用队列存储起来以备下一层遍历的时候再遍历。

套路：
向队列里填充初始值: (若干选择项, 待求结果如最短路径、最少步骤)
while 队列Q不空:
    (选择choice, step) = Q.popleft()
    if 满足返回条件: return step
    邻域 = choice符合边界条件的邻域（这个领域，可以看成是层的概念）
    for c in 邻域:
        标记c ==> 这个特别重要，防止死循环
        Q.append(c, 1+step)

收尾工作 (return xxx 或 后续判断)
"""


# 1091. 二进制矩阵中的最短路径
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if grid[0][0]==1 or grid[-1][-1]==1: return -1
        N = len(grid)

        # 对于给定的一个点，求不超过边界的邻域, 可以吧邻域看成【选择】
        def neighbors(r,c):
            return filter(
                lambda (x,y): 0<=x<N and 0<=y<N,
                [(r+1,c), (r+1,c-1), (r+1,c+1),(r,c+1),(r,c-1),(r-1,c),(r-1,c-1),(r-1,c+1)])
        
        q = collections.deque() # 准备好队列
        q.append((0, 0, 1)) # 队列填充初始值，其中前俩为坐标，最后一个”1“为步数
        while q:
            x, y, path = q.popleft()
            if x==y==N-1: return path # 一旦弹出的值符合条件，即返回
            for i,j in neighbors(x,y):
                if grid[i][j]==0:
                    q.append((i,j,1+path)) # 符合条件的邻域点以同等身份压入队列，注意此时待返回值path+1（标志这几个邻域在具有相同”时刻“,所以都是1+path）
                    grid[i][j] = 1 # 一个领域一旦被访问，则标注为1，以防止其他领域再次访问！！！因为走过了后续再走，就必定不是最短了
        return -1


# 994. 腐烂的橘子
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        from collections import deque 
        
        ## 队列填充初始值，带返回值d=0
        q = deque()
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    q.append((i,j,0))
        
        while q:
            i, j, d = q.popleft()
            for ni, nj in filter(lambda (x,y): 0<=x<m and 0<=y<n: [(i,j-1), (i,j+1), （i-1,j), (i+1,j)]):
                if grid[ni][nj] == 1:
                    grid[ni][nj] = 2 # 邻域一旦被访问，则被标记
                    q.append((ni,nj,1+d)) # 符合条件的邻域坐标和加一后的d以同等身份一同压入队列
                    
        if any(1 in row for row in grid): return -1
        return d


# 279. 完全平方数
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 方法一 BFS:
        q = collections.deque()
        q.append((n, 1))
        myset = set([n])
        while q:
            num, cnt = q.popleft()
            choices = [num - x*x for x in range(1, 1+int(num**0.5))] # 类比于第一题的”邻域“
            for choice in choices:
                if choice == 0: return cnt
                if choice not in myset: myset.add(choice); q.append((choice, 1+cnt)) # 注意这里，”邻域“被访问了，就要被标记！！！符合条件的邻域加入队列，并且结果cnt加一哦
        return -1
    
        # 方法二 动态规划：
        dp = [0]*(1+n)
        for i in range(1, 1+n):
            dp[i] = 1 + min(dp[i-j*j] for j in range(1, 1+int(i**0.5)))
        return dp[-1]
    
    
# 127. 单词接龙 (超时了)
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        
        def compare(s, t):
            cnt = 0
            for i,j in zip(s,t):
                if i==j: cnt += 1
            return len(s)-cnt==1
        
        def word_choices(word):
            r = []
            for w in wordList:
                if compare(word, w):
                    r.append(w)
            return r
        
        q = collections.deque()
        q.append((beginWord, 1))
        visited = set()
        while q:
            word, step = q.popleft()
            if word == endWord: return step
            choices = word_choices(word)
            for choice in choices:
                if choice not in visited: 
                    visited.add(choice)
                    q.append((choice, 1+step))
        return 0
        


#### 关于树的BFS，没啥难的

# 637. 二叉树的层平均值
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        
        cur, res = [root], 0
        while cur:
            nxt, res = [], sum(n.val for n in cur) / len(cur)
            for node in cur:
                if node.left: nxt.append(node.left)
                if node.right: nxt.append(node.right)
            cur = nxt
        return res

        
# 513. 找树左下角的值       
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        if not root: return []
        cur, res = [root], root.val
        while cur:
            nxt, res = [], cur[0].val
            for node in cur:
                if node.left: nxt.append(node.left)
                if node.right: nxt.append(node.right)
            cur = nxt
        return res
        