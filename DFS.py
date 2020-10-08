
# 面试题 08.10. 颜色填充
class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        originColor = image[sr][sc]
        if originColor == newColor: return image # 要判断新老色是否相同，相同就直接返回得了
        self.fill(image, sr, sc, originColor, newColor)
        return image

    def fill(self, image, i, j, o, t):
        m, n = len(image), len(image[0])
        if not (0 <= i < m and 0 <= j < n): return 
        if image[i][j] != o: return 
        image[i][j] = t
        self.fill(image, i-1, j, o, t)
        self.fill(image, i+1, j, o, t)
        self.fill(image, i, j-1, o, t)
        self.fill(image, i, j+1, o, t)


# 529. 扫雷游戏
class Solution(object):
    
    dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,1), (1,-1), (-1,1)]
    
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
                
        x, y = click
        if board[x][y] == 'M':
            board[x][y] = 'X'
            return board
        self.dfs(board, x, y)
        return board
    
    def inArea(self, board, x, y):
        return 0 <= x < len(board) and 0 <= y < len(board[0])
    
    def dfs(self, board, x, y): 
        if self.inArea(board, x, y) and board[x][y] == 'E':
            cnt = 0
            for dir in self.dirs:
                dx, dy = dir
                if self.inArea(board, x+dx, y+dy) and (board[x+dx][y+dy] == 'M' or board[x+dx][y+dy] == 'X'):
                    cnt += 1
            if cnt > 0:
                board[x][y] = str(cnt)
                return
            board[x][y] = 'B'
            
            for dir in self.dirs:
                dx, dy = dir
                if self.inArea(board, x+dx, y+dy) and board[x+dx][y+dy] == 'E':
                    self.dfs(board, x+dx, y+dy)


# 695. 岛屿的最大面积
class Solution(object):
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        ma = 0
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                ma = max(ma, self.area(grid, x, y))
        return ma
    
    def area(self, grid, x, y):
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]): 
            if grid[x][y] == 1: 
                grid[x][y] = 0 # 这个很重要；作为访问过得标记，避免重复访问
                return 1 + self.area(grid, x-1, y) + self.area(grid, x+1, y) + self.area(grid, x, y-1) + self.area(grid, x, y+1)
            return 0
        else:
            return 0


# 200. 岛屿数量
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        
        def F(x, y):
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                if grid[x][y] == '1':
                    grid[x][y] = '0'
                    F(x-1,y); F(x+1,y); F(x,y-1); F(x, y+1)
                else:
                    return
            return 
        
        cnt = 0
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == '1':
                    cnt += 1
                    F(x, y)
        return cnt
                    

# 1254. 统计封闭岛屿的数目(题目的意思是找到不与边界相接的陆地)                   
class Solution(object):
    def closedIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        # 题解
        # https://leetcode-cn.com/problems/number-of-closed-islands/solution/dian-xing-dao-yu-ti-dfsjie-ti-xi-jie-by-happyfire/
        
        
        # 返回的是一个bool值；若是陆地，则消消乐，"顺便"看下当前坐标所连接的陆地是否和边界有接触
        def touch(x, y):
            if not (0 <= x < len(grid) and 0 <= y < len(grid[0])): return True
            if grid[x][y] == 1: return False
            
            # 如果遇到陆地(即0), 先做访问过的标记置1，然后递归
            grid[x][y] = 1
            u = touch(x-1, y)
            b = touch(x+1, y)
            l = touch(x, y-1)
            r = touch(x, y+1)
            return u or b or l or r 
        
        
        cnt = 0
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == 0:
                    cnt += not touch(x, y)
        return cnt
            
            
            
                
            
                
            
                