"""
背包问题的总结
https://leetcode-cn.com/problems/combination-sum-iv/solution/xi-wang-yong-yi-chong-gui-lu-gao-ding-bei-bao-wen-/
"""



# 70. 爬楼梯
class Solution(object):
    def climbStairs(self, n):
        if n == 0 or n == 1:return 1
        pre, cur = 1, 1
        for _ in range(2, n+1):
            pre, cur = cur, pre + cur
        return cur
    
# 198. 打家劫舍
class Solution(object):
    def rob(self, nums):
       # F(n) = max(A[n] + F(n-2), F(n-1))
       pre, cur = 0, 0
       for num in nums:
           pre, cur = cur, max(cur, pre + num)
       return cur


class Solution(object):
    def rob(self, nums):
        if len(nums) == 1:
            return nums[0]
        
        def F(A):
            pre, cur = 0, 0
            for a in A:
                pre, cur = cur, max(pre + a, cur)
            return cur
        
        return max([F(nums[:-1]), F(nums[1:])])
    
# 413. 等差数列划分
# https://leetcode-cn.com/problems/arithmetic-slices/solution/chang-yong-tao-lu-jie-jue-dong-tai-gui-hua-by-lu-c/
class Solution(object):
    def numberOfArithmeticSlices(self, A):
        dp = [0] * len(A)
        for i in range(2, len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                dp[i] = 1 + dp[i-1]
        return sum(dp)


class Solution(object):
    memo = {}
    def rob(self, root):
        if not root: return 0
        if self.memo.get(root): return self.memo.get(root)
        
        do = root.val
        if root.left: do += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right: do += self.rob(root.right.left) + self.rob(root.right.right)
        not_do = self.rob(root.left) + self.rob(root.right)
        
        r = max(do, not_do)
        self.memo[root] = r
        
        return r
    
# 120 三角形最小路径和     
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        A = triangle
        for i in range(len(A)):
            for j in range(len(A[0])):
                if i == j == 0: continue
                elif j == 0: A[i][j] += A[i-1][0] # 处于当前行的开始
                elif j == i: A[i][j] += A[i-1][i-1] # 处于当前行的末尾
                else: A[i][j] += min([A[i-1][j-1], A[i-1][j]])
        return min(A[-1])

# 最长公共子串 (注意不是子序列)
class Solution:
    def LCS(self , s1 , s2 ):
        # write code here
        m, n = len(s1), len(s2)
        dp = [[0]*(1+n) for _ in range(1+m)]
        lcs = 0
        for i in range(1,1+m):
            for j in range(1,1+n):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                    lcs = max(lcs, dp[i][j])
        return lcs

# 求最长公共子序列LCS的具体子序列
class Solution:
    def LCS(s1, s2):
        # 计算LCS的长度
        n = len(s1)
        m = len(s2)
        dp = [[0]*(m+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for j in range(1, m+1):
                if s1[i-1] == s2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
                else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        if dp[-1][-1] == 0: return -1
        i, j = n, m; s = ''
        while i > 0 and j > 0:
            if s1[i-1] == s2[j-1]:
                s = s2[j-1] + s
                i -= 1
                j -= 1
                continue
            else:
                if dp[i][j-1] >= dp[i-1][j]: j -= 1
                else: i -= 1
        return s
    
# 最长上升子序列
class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        A = nums
        dp = [1] * len(A)
        for i in range(len(nums)):
            for j in range(i):
                if A[i] > A[j]:
                    dp[i] = max(dp[i], 1 + dp[j])
        return max(dp)


# 类似于最长上升子序列的题
class Solution(object):
    def findLongestChain(self, pairs):
        """
        :type pairs: List[List[int]]
        :rtype: int
        """
        A = pairs
        A.sort(key=lambda x: x[1])
        
        dp = [1] * len(A)
        for i in range(1, len(dp)):
            for j in range(i):
                if A[i][0] > A[j][1]:
                    dp[i] = max(dp[i], 1 + dp[j])
        return max(dp)
    
    
# 152. 乘积最大子数组
# https://leetcode-cn.com/problems/maximum-product-subarray/solution/duo-chong-si-lu-qiu-jie-by-powcai-3/
class Solution(object):
    def maxProduct(self, nums):
        ans = curmin = curmax = nums[0]
        for n in nums[1:]:
            tmp = curmin
            curmin = min(n, n*curmin, n*curmax)
            curmax = max(n, n*tmp, n*curmax)
            ans = max(ans, curmax)
        return ans

# 343. 整数拆分
class Solution:
    def integerBreak(self, n: int) -> int:
        dp = [0] * (n+1)
        dp[1] = dp[2] = 1
        for i in range(3, n+1):
            dp[i] = max(max(dp[i-j], i-j) * j for j in range(i)) # 这里的max(dp[i-j], i-j)容易忽略
        return dp[n]


# 牛客 子数组累加最大和  
class Solution:
    def maxsumofSubarray(self , arr ):
        # write code here
        ma = premax = arr[0]
        for n in arr[1:]: 
            if premax > 0:
                premax += n
            ma = max(premax, ma)
        return ma

# 牛客 子数组累加最大和(原地修改)
class Solution:
    def maxsumofSubarray(self , arr ):
        for i in range(1, len(arr)):
            arr[i] += max(0, arr[i-1])
        return max(arr)

# 最长摆动子序列
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        A = nums
        if not A or len(A) == 0: return 0
        up, down = 1, 1
        for i in range(1, len(A)):
            if A[i] > A[i-1]: up = down + 1
            elif A[i] < A[i-1]: down = up + 1
        return max(up, down)
            
    
# 91. Decode Ways (Medium) (这里用了回溯算法，其实应该用动态规划复杂度要低，但是前者好理解)
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        self.count = 0
        def F(l, remain):
            if l == len(s):
                self.count += 1
                return 
            for i in range(1, len(remain)+1):
                prefix = remain[:i]
                if i > 3 or prefix[0] == '0' or int(prefix) > 26: continue
                F(l + i, remain[i:])
        F(0, s)
        return self.count


# 91. Decode Ways (Medium) # 很差的解法，条件太细了
class Solution(object):
    def numDecodings(self, s):
        
        if s[0] == '0': return 0
        dp = [0] * (len(s) + 1)
        dp[0], dp[1] = 1, 1
        for i in range(1, len(s)):
            if s[i] == '0':
                if s[i-1] == '1' or s[i-1] == '2': dp[i+1] = dp[i-1]
                else: return 0
            else:
                if s[i-1] == '1' or (s[i-1] == '2' and int(s[i]) <= 6): dp[i+1] = dp[i-1] + dp[i]
                else: dp[i+1] = dp[i]
        return dp[-1]

# 91. 解码方法
# 优秀解法
# https://leetcode.cn/problems/decode-ways/solution/gong-shui-san-xie-gen-ju-shu-ju-fan-wei-ug3dd/
class Solution:
    def numDecodings(self, s: str) -> int:
        s = ' ' + s
        dp = [0] * len(s)
        dp[0] = 1
        for i in range(1, len(s)):
            if 1 <= int(s[i]) <= 9:
                dp[i] = dp[i-1]
            if 10 <= int(s[i-1:i+1]) <= 26:
                dp[i] += dp[i-2]
        return dp[-1]
    
    
# 最长回文子序列
class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        N = len(s)
        dp = [[0] * N for _ in range(N)]
        
        for i in range(N): dp[i][i] = 1
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                if s[i] == s[j]: dp[i][j] = 2 + dp[i+1][j-1]
                else: dp[i][j] = max(dp[i][j-1], dp[i+1][j])
        return dp[0][N-1]


class Solution:
    def minnum(self, N):
        x = 1024 - N
        dp = [0] * (x + 1)
        for i in range(1, x+1):
            dp[i] = 1 + min(dp[i-choice] for choice in [1, 4, 16, 64] if choice <= i)
        return dp[-1]
    

class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        
        def F(x, start):
            if start == len(nums):
                if x == S:
                    return 1
                else:
                    return 0
            return F(x + nums[start], start+1) + F(x - nums[start], start + 1)


# 518. 零钱兑换 II(完全背包+组合问题；数组中的元素可重复使用，nums放在外循环，target在内循环。且内循环正序)
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        
        可以吧dp数组设置为1维
        把coins物品选择设置为外层循环，一层一层更新dp数组，对每一个dp元素进行累加, 是一种累和操作
        """
        if amount == 0: return 1
        if not coins: reuturn 0
        dp = [0] * (1+amount)
        dp[0] = 1
        for coin in coins:
            for i in range(coin, 1+amount):
                dp[i] += dp[i-coin]
        return dp[-1]
      
             
# 322. 零钱兑换 
class Solution(object):
    def coinChange(self, coins, amount):
        """
        "(完全背包问题，即数组中的元素可重复使用，nums放在外循环，target在内循环。且内循环正序。)"
        这里并没有采用这一原则！
        """
        dp = [1 + amount] * (1 + amount) # 初始化一个不可能的值，若计算结果仍然未变，则说明没有合适的组合
        dp[0] = 0 
        
        for i in range(1, 1 + amount):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], 1 + dp[i - c])
        if dp[-1] == 1 + amount: return -1
        return dp[-1]


# 322. 零钱兑换 (二维遍历，很标准的流程，没有上面的技巧)
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[i][j] = 前i个钱币，背包容量为j，可以获得最少硬币个数
        V = amount
        N = len(coins)
        dp = [[1 + amount] * (1 + V) for _ in range(1 + N)] # 初始化1+amount,相当于初始化最大值,1+amount为不可可能的最大值，便于后续取min
        for i in range(1 + N):
            dp[i][0] = 0

        for i in range(1, 1 + N):
            for j in range(1, 1 + V):
                if j < coins[i - 1]:
                    dp[i][j] = dp[i - 1][j] # 背包装不下
                else:
                    dp[i][j] = min(dp[i - 1][j], 1 + dp[i][j - coins[i - 1]]) # 1 + dp[i][j - coins[i - 1]] 注意这里是dp[i][...]可以重复放
        return dp[-1][-1] if dp[-1][-1] != 1 + amount else -1


# 377. 组合总和 Ⅳ(如果组合问题需考虑元素之间的顺序，需将target放在外循环，将nums放在内循环。)
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        
        https://leetcode-cn.com/problems/combination-sum-iv/solution/dong-tai-gui-hua-python-dai-ma-by-liweiwei1419/
        
         /**
        * 这里状态定义就是题目要求的，并不难，状态转移方程要动点脑子，也不难：
        * 状态转移方程：dp[i]= dp[i - nums[0]] + dp[i - nums[1]] + dp[i - nums[2]] + ... （当 [] 里面的数 >= 0）
        * 特别注意：dp[0] = 1，表示，如果那个硬币的面值刚刚好等于需要凑出的价值，这个就成为 1 种组合方案
        * 再举一个具体的例子：nums=[1, 3, 4], target=7;
        * dp[7] = dp[6] + dp[4] + dp[3]
        * 即：7 的组合数可以由三部分组成，1 和 dp[6]，3 和 dp[4], 4 和dp[3];
        *
        * @param nums
        * @param target
        * @return
        */

        作者：liweiwei1419
        链接：https://leetcode-cn.com/problems/combination-sum-iv/solution/dong-tai-gui-hua-python-dai-ma-by-liweiwei1419/
        来源：力扣（LeetCode）
        著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
        
        """
        dp = [0] * (1 + target)
        dp[0] = 1 
        for i in range(1, 1+target):
            for num in nums:
                if i >= num:
                    dp[i] += dp[i - num]
        return dp[-1]


# 单词切割 完全背包问题(考虑顺序)
class Solution:
    def wordBreak(self, s, wordDict): 
        dp = [True] + [False] * len(s)
        for i in range(1, len(dp)):
            for w in wordDict:
                if i >= len(w) and s[i-len(w) : i] == w:
                    dp[i] = dp[i] or dp[i - len(w)]
        return dp[-1]
 

# # 416. 分割等和子集 0-1背包问题 (采用二维数组，正序)
# class Solution(object):
#     def canPartition(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: bool
#         """
#         s = sum(nums) % 2
#         if s == 1: return False
        
#         dp = [[False] * (1 + s) for _ in range(1 + len(nums))]
#         for i in range(1 + len(nums)): dp[i][0] = True
#         for i in range(1, 1 + len(nums)):
#             for j in range(1, 1 + s):
#                 if j < nums[i-1]: dp[i][j] = dp[i-1][j] # 都是用到了 i-1层
#                 else: dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]] # 都是用到了 i-1层
#         return dp[-1][-1]


# 416. 分割等和子集: 数组是否可以分割为两部分使得两部分的和相等 0-1背包问题 (采用一维数组，倒叙)
# 0-1背包问题，nums(物品列表)在外层循环; target在内层循环，为倒叙 
class Solution(object):
    def canPartition(self, nums):
        s = sum(nums)
        if s % 2 == 1:
            return False
        
        target = s / 2
        dp = [True] + [False] * target # dp[0]=True; 因为背包容量是0，任何物品都可以满足，不用装
        
        for num in nums:
            for j in range(len(dp), num-1, -1):
                dp[j] = dp[j] or dp[j - num] # (放 or 不放)
        return dp[-1]
    
    
# # 474. 一和零 (0-1背包的多为问题，整理思路，3维数组，正序，非最优解)
# class Solution:
#     # https://leetcode-cn.com/problems/ones-and-zeroes/solution/dong-tai-gui-hua-zhuan-huan-wei-0-1-bei-bao-wen-ti/
#     def findMaxForm(self, strs, m, n):
#         import numpy as np
#         l = len(strs)
#         dp = np.zeros((1+l, 1+m, 1+n), dtype=np.int32).tolist()
#         for i in range(1, l + 1):
#             for j in range(m + 1):
#                 for k in range(n + 1):
#                     ones = strs[i - 1].count("1")
#                     zeros = strs[i - 1].count("0")
#                     dp[i][j][k] = dp[i - 1][j][k] # 只有我的背包有容量的时候才进行更新，没有容量的时候要此时 dp[i][j][k]=dp[i-1][j][k]
#                     if j >= zeros and k >= ones:  # 遍历到i时，背包有空余容量装入此时的 0s 1s
#                         dp[i][j][k] = max(dp[i - 1][j - zeros][k - ones] + 1, dp[i-1][j][k])
        # return dp[-1][-1][-1]

# 474. 一和零 (0-1背包问题；降维后二维数组，倒叙)
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        
        dp = [[0] * (n+1) for _ in range(m+1)]
        for s in strs:
            one = s.count('1')
            zero = s.count('0')
            
            for i in range(m, zero-1, -1):
                for j in range(n, one-1, -1):
                    dp[i][j] = max(dp[i][j], 1 + dp[i-zero][j-one])
        return dp[-1][-1]


# 121. 买卖股票的最佳时机
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        min_price = prices[0]
        max_profit = -float('inf')
        
        for p in prices:
            max_profit = max(max_profit, p - min_price)
            min_price = min(min_price, p)
        return max_profit


# 121.买卖股票的最佳时机 (动态规划方法一)
class Solution(object):
    def maxProfit(self, prices):
        
        dp = [[0,0] * len(prices)]
        
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
        return dp[-1][0]


# 121.买卖股票的最佳时机 (动态规划方法二，空间复杂度O(1))
class Solution(object):
    def maxProfit(self, prices):
        dp_0, dp_1 = 0, -prices[0]
        
        for i in range(1, len(prices)):
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, -prices[i])
        return dp_0


# 122. 买卖股票的最佳时机 II k = +infinity    
class Solution(object):
    def maxProfit(self, prices):
        dp_0, dp_1 = 0, -prices[0]
        
        for i in range(1, len(prices)):
            tmp = dp_0 # 前一天买卖所产生的利润
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, tmp - prices[i])
        return dp_0


# 309. 买卖股票的最佳时机 III k = +infinity with cooldown   
class Solution(object):
    def maxProfit_with_cool(self, prices):
        dp_pre_0, dp_0, dp_1 = 0, 0, -prices[0]
        
        for i in range(1, len(prices)):
            tmp= dp_0
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, dp_pre_0 - prices[i])
            dp_pre_0 = tmp
        return dp_0
    

# 714. 买卖股票有费用的 k = +infinity with fee   
class Solution(object):
    def maxProfit_with_fee(self, prices, fee):
        dp_0, dp_1 = 0, -prices[0] - fee
        
        for i in range(1, len(prices)):
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, dp_0 - prices[i] - fee)
        return dp_0


# 123. 买卖股票的最佳时机 k = 2            
class Solution(object):
    def maxProfit(self, prices):
        
        import numpy as np
        
        K = 2
        L = len(prices)
        
        dp = np.zeros((L, K+1, 2)).tolist()
        
        for i in range(L):
            for k in range(1， K):
                if i == 0:
                    dp[0][k][0] = 0
                    dp[0][k][1] = -prices[i]
                    continue
                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        return max(dp[-1][K][0], dp[-1][K][1])



class Solution(object):
    def maxProfit(self, k, prices):
        
        if not prices:
            return 0
        
        import numpy as np
        
        kk = k
        L = len(prices)
        K = int(L // 2)
        
        if kk < K:
            K = kk
        else:
            dp_0, dp_1 = 0, -prices[0]
            for i in range(1, L):
                tmp = dp_0
                dp_0 = max(dp_0, dp_1 + prices[i])
                dp_1 = max(dp_1, tmp - prices[i])
            return dp_0                
            
   
        dp = np.zeros((L, K+1, 2), dtype=np.int32).tolist()
        for i in range(L):
            for k in range(K, 0, -1):
                if i == 0:
                    dp[0][k][0] = 0
                    dp[0][k][1] = -prices[i]
                    continue
                dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
                dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
        return max(dp[-1][K][0], dp[-1][K][1])


# 714 
class Solution(object):
    def maxProfit(self, prices, fee):
        dp_0, dp_1 = 0, -prices[0] - fee
        
        for i in range(1, len(prices)):
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, dp_0 - fee - prices[i])
        return dp_0

            

# 583 编辑距离            
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        import numpy as np
        s = word1; t = word2; M = len(s); N = len(t)
        dp = np.zeros((M+1, N+1)).tolist()
        for j in range(N+1): dp[0][j] = j
        for i in range(M+1): dp[i][0] = i
        for i in range(1, 1+M):
            for j in range(1, 1+N):
                if s[i-1] == t[j-1]: dp[i][j] = dp[i-1][j-1]
                else: dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[-1][-1]


# 221. 最大正方形
# https://leetcode-cn.com/problems/maximal-square/solution/dong-tai-gui-hua-han-kong-jian-you-hua-221-zui-da-/
# # dp[i][j]表示以matrix[i][j]为右下角的顶点的可以组成的最大正方形的边长
class Solution:
    def maximalSquare(self, mat):
        m = len(mat)
        if m == 0: return 0
        n = len(mat[0])
        ans = 0
        dp = [(1+n)*[0] for _ in range(1+m)] # 在原来的数组mat上包了一层边儿
        for i in range(1, 1+m):
            for j in range(1, 1+n):
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) if mat[i-1][j-1] == '1' else 0 # 注意mat[i-1][j-1]，因为包边儿了
                ans = max(ans, dp[i][j])
        return ans**2


# 354. 俄罗斯套娃信封问题
# 按照宽度升序后，按照h降序，因为宽度一样时不能嵌套，索性就把h大的移动到前面（降序）
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        if not envelopes: return 0
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        arr = [x[1] for x in envelopes]
        dp = [1]*len(arr)
        for i in range(len(arr)):
            for j in range(i):
                if arr[i] > arr[j]: 
                    dp[i] = max(dp[i], 1+dp[j])
        return max(dp)
        
        


