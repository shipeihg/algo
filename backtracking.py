

"""
同类型题解
https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-by-powcai-2/
"""


"""
回溯算法套路
回溯算法一般是用来返回排列组合的结果的，这一点切记！




"""


# 46. 全排列
class Solution(object):
    def permute(self, nums):
        if not nums:
            return []
        
        r = []
        def F(path):
            if len(path) == len(nums):
                r.append(path)
                return 

            for i in range(len(nums)):
                if nums[i] not in path:
                    F(path + [nums[i]])
        F([])
        return r



class Solution:
    def subsets(self, nums):
        r = []
        s = set()
        def F(start, path):
            if tuple(path) not in s:
                s.add(tuple(path))
                r.append(path)
                for i in range(start, len(nums)):
                    F(start, path + [nums[i]])
        F(0, [])
        return r
    
    
class Solution(object):
    def combinationSum2(self, candidates, target):
        
        candidates.sort()
        
        if not candidates:
            return []
        if min(candidates) > target:
            return []
        
        r = []
        def F(path, target, nums):
            if path not in r:
                if target == 0:
                    r.append(path)
                    return 
                if target <  0:
                    return
                
                for i in range(len(nums)):
                    if nums[i] > target:
                        return
                    F(path + [nums[i]], target - nums[i], nums[i+1:])
        F([], target, candidates)
        return r


class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        
        nums = range(1, n+1)
        
        r = []
        def F(path, nums, depth):
            if depth == k:
                r.append(path)
                return
            for i in range(len(nums)):
                F(path + [nums[i]], nums[i+1:], 1 + depth)            
        F([], nums, 0)
        return r
    
    
class Solution(object):
    def solveNQueens(self, n):
        r = []
        def F(path):
            if len(path) == n:
                r.append(path)
                return

            for col in range(n):
                if col in path:
                    continue
                if  path and any(abs(len(path)-i)==abs(col-j) for i,j in enumerate(path)):
                    continue
                F(path + [col])
        F([])
        return self.transform(r, n)
        
    def transform(self, paths, n):
        rr = []
        for path in paths:
            arr = ['.' * n for _ in range(n)]
            for i,j in enumerate(path):
                arr[i] = arr[i][:j] + 'Q' + arr[i][j+1:]
            rr.append(arr)
        return rr
            
        
class Solution(object):
    def partition(self, s):
        
        r = []
        def F(path, string, length):
            if length > len(s):
                return
            if length == len(s):
                r.append(path)
                return
            
            for i in range(1, len(string)+1):
                if self.isSymetric(string[:i]):
                    F(path + [string[:i]], string[i:], len(string[:i]) + length)
        F([], s, 0)    
        return r
           
    def isSymetric(self, s):
        i, j = 0, len(s) - 1
        while i < j: 
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
    
    
class Solution(object):
    def solveSudoku(self, board):
        """
        37. 解数独
        https://leetcode-cn.com/problems/sudoku-solver/solution/pythonsethui-su-chao-guo-95-by-mai-mai-mai-mai-zi/
        """
        
        rows = [set(range(1, 10)) for _ in range(9)]
        cols = [set(range(1, 10)) for _ in range(9)]
        blocks = [set(range(1, 10)) for _ in range(9)]
        
        empty = []
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    val = int(board[i][j])
                    rows[i].remove(val)
                    cols[j].remove(val)
                    blocks[(i//3)*3 + j//3].remove(val)
                else:
                    empty.append((i,j))
        
        def F(depth=0):
            if depth == len(empty):
                return True
            
            i, j = empty[depth]           
            k = (i//3)*3 + j//3
            for val in rows[i] & cols[j] & blocks[k]: # 层遍历树的当前深度的所有节点
                rows[i].remove(val)
                cols[j].remove(val)
                blocks[k].remove(val)
                board[i][j] = str(val)
                if F(1 + depth):
                    return True
                rows[i].add(val)
                cols[j].add(val)
                blocks[k].add(val)
            return False
        F()
        
        
            
        