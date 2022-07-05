

"""
同类型题解
https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-by-powcai-2/
"""


"""
回溯算法套路
回溯算法一般是用来返回排列组合的结果的，这一点切记！


一句话总结：在当前状态下，遍历所有接下来的分支！！！！

1.永远记住俩个要素：【路径 + 选择】(理解为：当前的路径+剩余的选择)
2.若数组里有重复项，记得先排序
3.若结果中要求不能含有重复项，记得递归中用 if path not in res 来排除，虽然线性复杂度，但是好理解
4.【最重要的一点】: 递归函数写成 F(path当前路径, choices剩余的选择), 相当于每一个函数都维护着一对(path, choices), 因为choices不是全局变量，所以根本不用模板中的撤销选择，但这样做的代价是增加内存，好处是写着方便，不用担心撤销选择，比较无脑


def solution():
    存放最终结果的列表 res = []
    def backtrack(路径path, 选择choice):
        if len(path)==目标长度:
            res.append(path)
            return
        for 选择 in choices:
            backtrack(path+当前选择, 排除choice后另外新的选择)

    backtrack([], 0)
    return res
"""


# 93. 复原IP地址
class Solution:
    def restoreIpAddresses(self , s ):
        # write code here
        ans = []
        def F(path, L, choices):
            if len(path) == 4 and L == len(s):
                ans.append('.'.join(path))
                return
            for i in range(1, len(choices)+1):
                prefix = choices[:i]
                if i > 3: continue 
                if i > 1 and prefix[0] == '0': continue 
                if int(prefix) > 255: continue
                F(path + [prefix], L + i, choices[i:])
        F([], 0, s)
        return ans


# 46. 全排列
class Solution(object):
    def permute(self, nums):
        if not nums: return []
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


# 47. 全排列 II 
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        ans = []
        def F(path, choices):
            if len(path)==len(nums) and path not in ans:
                ans.append(path)
                return
            for i in range(len(choices)):
                F(path + [choices[i]], choices[:i] + choices[i+1:])
        F([],nums)
        return ans
        

class Solution:
    def subsets(self , A ):
        # write code here
        ans = []
        def F(path, choices):
            if len(path) <= len(A):
                ans.append(path)
            
            for i in range(len(choices)):
                F(path + [choices[i]], choices[i+1:])
        F([], A)
        return ans

# 90. 子集 II
class Solution(object):
    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort() # 先排序！！！ 因为nums中含有重复项，且要求结果不能有重复的组合
        ans = []
        def F(path, choices):
            if len(path)<=len(nums) and path not in ans:
                ans.append(path)
            for i in range(len(choices)):
                F(path+[choices[i]], choices[i+1:])
        F([],nums)
        return ans

    
#39. 组合总和 (candidates 中的数字可以无限制重复被选取。)
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort() # 排序，从小的开始遍历
        ans = []
        def F(path, choices, target):
            if target == 0: ans.append(path)
            for i in range(len(choices)):
                if target < choices[i]: break # 提前结束搜寻
                F(path+[choices[i]], choices[i:], target-choices[i]) # choices[i:]表示将包括i及其以后的数都考虑在内
        F([], candidates, target)
        return ans
    
# 40. 组合总和 II
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        ans = []
        def F(path, choices, target):
            if path not in ans: # 要求结果不能有重复项; 加上加句话是因为candidates有重复项
                if target == 0: ans.append(path)
                for i in range(len(choices)):
                    if target < choices[i]: return
                    F(path + [choices[i]], choices[i+1:], target-choices[i]) # 注意区别 choices[i+1:]
        F([], candidates, target)
        return ans


# 216. 组合总和 III
class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        ans = []
        def F(path, choices, target):
            if target == 0 and len(path) == k: ans.append(path); return
            if target != 0 and len(path) == k: return
            if target < 0: return 
                
            for i in range(len(choices)):
                F(path + [choices[i]], choices[i+1:], target - choices[i])
        F([], range(1,10), n)
        return ans


class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        
        ans = []
        def F(path, choices):
            if len(path) == k: 
                ans.append(path)
            for i in range(len(choices)):
                F(path + [choices[i]], choices[i+1:])
        F([], range(1,n+1))
        return ans
        
# 51. N皇后
class Solution(object):
    def solveNQueens(self, n):
        r = []
        def F(path):
            if len(path) == n:
                r.append(path)
                return

            for col in range(n): # 选择
                if col in path: # 垂直方向不能共线
                    continue
                if  path and any(abs(len(path)-i)==abs(col-j) for i,j in enumerate(path)): # 对角方向不能共线
                    continue
                F(path + [col]) # 接下来的选择完全可以通过path来排除
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
        
# 131. Palindrome Partitioning (Medium)      
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
            for val in rows[i] & cols[j] & blocks[k]: # 层遍历树的当前深度的所有节点；在(i,j)位置下可供选择的数字
                rows[i].remove(val)
                cols[j].remove(val)
                blocks[k].remove(val) # 排除val，实际上是为F(1+depth)做选择
                board[i][j] = str(val)
                if F(1 + depth):
                    return True # 这个就厉害了，数独可能有多个解，F返回bool，一旦为True就退出！
                rows[i].add(val) # 上面做完选择要撤销选择
                cols[j].add(val)
                blocks[k].add(val)
            return False
        F()
        
        
# 17. 电话号码的字母组合
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        
        if not digits: return []
        lookup = {
            "2":"abc",
            "3":"def",
            "4":"ghi",
            "5":"jkl",
            "6":"mno",
            "7":"pqrs",
            "8":"tuv",
            "9":"wxyz"
        }
        
        ans = []
        def F(i, path): 
            if i == len(digits):
                ans.append(path)
                return
            for char in lookup[digits[i]]:
                F(i+1, path + char)
        F(0,'')
        return ans
      
                
# 79. 单词搜索            
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board or not board[0]:
            return False
        A = board
        m, n = len(A), len(A[0])

        def backtrack(i, j, index):
            if A[i][j] != word[index]: return False
            if index == len(word) - 1: return True
            tmp = A[i][j]
            A[i][j] = 0
            b = False
            for (ii, jj) in [(i - 1, j), (i + 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= ii < m and 0 <= jj < n:
                    b = b or backtrack(ii, jj, 1 + index)
            A[i][j] = tmp
            if b:
                return True
            else:
                return False

        for i in range(m):
            for j in range(n):
                if backtrack(i, j, 0):
                    return True
        return False

# 91. 解码方法 (这道题超时，思路其实是DP，但是我没想出来)
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if not s or s == '0': return 0
        
        L = len(s); self.cnt=0
        def F(length, s):
            if length == L: self.cnt += 1
            i=0
            while i < len(s):
                if int(s[:i+1]) > 26: continue # 超过26不能解码
                if i < len(s)-1 and s[i+1] == '0': continue # 当前字符的下一个字符不可以是0
                if s[i] == '0': continue # 当前字符不能是0，否则会变成 03，他就不是一个数了
                F(length + 1 + i, s[i+1:])
                i += 1
        F(0, s)
        return self.cnt


# 494. 目标和 (超出时间限制， 正确思路应该是DP)
class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        # 回溯1
        self.cnt = 0
        def F(sum, i):
            if i == len(nums):
                if sum == S:
                    self.cnt += 1
                return # 一定要在 i==len(nums) 时就停止 --> return
            F(sum + nums[i], 1+i)
            F(sum - nums[i], 1+i)
        F(0, 0)
        return self.cnt
        
        # 回溯2
        def G(path, i):
            if len(path) == len(nums):
                if sum(path) == S:
                    self.cnt += 1
                return 
            G(path + [num[i]], i+1)
            G(path + [-nums[i]],i+1)
        G([], 0)
        retur self.cnt
    
            

            