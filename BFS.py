

# 994. 腐烂的橘子
class Solution(object):
    def orangesRotting(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        from collections import deque 
        
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
                    grid[ni][nj] = 2
                    q.append((ni,nj,1+d))
                    
        if any(1 in row for row in grid): return -1
        return d