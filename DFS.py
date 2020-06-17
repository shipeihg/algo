
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
        
        def fill(image, x, y, originColor, newColor):            
            if not (0<=x<len(image) and 0<=y<len(image[0])): return
            if image[x][y] != originColor: return
            if image[x][y] == -1: return # 已经被访问过了，防止重复访问
            
            image[x][y] = -1 # 作为访问过得标记，避免重复访问
            fill(image, x+1, y, originColor, newColor)
            fill(image, x-1, y, originColor, newColor)
            fill(image, x, y-1, originColor, newColor)
            fill(image, x, y+1, originColor, newColor)
            image[x][y] = newColor
        
        
        originColor = image[sr][sc]
        fill(image, sr, sc, originColor, newColor)
        return image


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
                    
