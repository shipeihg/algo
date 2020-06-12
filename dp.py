class Solution(object):
    def climbStairs(self, n):
        if n == 0 or n == 1:
            return 1
        
        pre, cur = 1, 1
        for _ in range(2, n+1):
            pre, cur = cur, pre + cur
        return cur
    

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


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    
    memo = {}
    
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        if not root:
            return 0
        
        if self.memo.get(root):
            return self.memo.get(root)
        
        # do = (root.val + 
        #       (self.rob(root.left.left) + self.rob(root.left.right)) if root.left else 0 +
        #       (self.rob(root.right.left) + self.rob(root.right.right)) if root.right else 0
        # )
        
        do = root.val
        if root.left:
            do += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right:
            do += self.rob(root.right.left) + self.rob(root.right.right)
        
        not_do = self.rob(root.left) + self.rob(root.right)
        
        r = max([do, not_do])
        
        self.memo[root] = r
        
        return r
    
           
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        
        A = triangle
        
        for i in range(len(A)):
            for j in range(len(A[0])):
                if i == j == 0:
                    continue
                elif j == 0:
                    A[i][j] += A[i-1][j]
                elif j == i:
                    A[i][j] += A[i-1][i-1]
                else:
                    A[i][j] += min([A[i-1][j-1], A[i-1][j]])
        return min(A[-1])
    

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
    

class Solution(object):
    def numberOfArithmeticSlices(self, A):
        
        dp = [0] * len(A)
        for i in range(2, len(A)):
            if A[i]-A[i-1] == A[i-1]-A[i-2]:
                dp[i] = 1 + dp[i-1]
        return sum(dp)
    

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
            
            for i in range(1, len(remain)+1):
                prefix = remain[:i]
                if prefix[0] == '0' or int(prefix) > 26:
                    continue
                F(l + len(remain[:i]), remain[i:])

        F(0, s)
        return self.count

ss = Solution()
print ss.numDecodings('0')


class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        if s[0] == '0':
            return 0
        
        dp = [0] * (len(s) + 1)
        dp[0], dp[1] = 1, 1
        for i in range(1, len(s)):
            if s[i] == '0':
                if s[i-1] == '1' or s[i-1] == '2':
                    dp[i+1] = dp[i-1]
                else:
                    return 0
            else:
                if s[i-1] == '1' or (s[i-1] == '2' and int(s[i]) <= 6):
                    dp[i+1] = dp[i-1] + dp[i]
                else:
                    dp[i+1] = dp[i]
        return dp[-1]


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
    

class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        A = nums
        
        if not A or len(A) == 0:
            return 0
        
        up, down = 1, 1
        for i in range(1, len(A)):
            if A[i] > A[i-1]:
                up = down + 1
            elif A[i] < A[i-1]:
                down = up + 1
        return max(up, down)
    

class Solution(object):
    def longestPalindromeSubseq(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        N = len(s)
        dp = [[0] * N for _ in range(N)]
        
        for i in range(N):
            dp[i][i] = 1
        
        for i in range(N-1, -1, -1):
            for j in range(i+1, N):
                if s[i] == s[j]:
                    dp[i][j] = 2 + dp[i+1][j-1]
                else:
                    dp[i][j] = max(dp[i][j-1], dp[i+1][j])
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
        
        # self.r = 0
        # def F(x, start):
        #     if start == len(nums):
        #         if x == S:
        #             self.r += 1
        #         return
        #     F(x+nums[start], start+1)
        #     F(x-nums[start], start+1)
        # F(0, 0)
        # return self.r
        
        def F(x, start):
            if start == len(nums):
                if x == S:
                    return 1
                else:
                    return 0
            
            return F(x + nums[start], start+1) + F(x - nums[start], start + 1)
        
class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        
        coins.insert(0, 0)
        A = coins
        memo = {}
        
        def F(i, j):
            
            if memo.get((i,j)):
                return memo[(i,j)]
            
            if i == j == 0:
                x = 1
            if i == 0 and j > 0:
                x = 0
            if i == 1 and j % A[1] == 0:
                x = 1
            if i == 1 and j % A[i] != 0:
                x = 0
            if i > 1 and j < A[i]:
                x = F(i-1, j)
            if i > 1 and j >= A[i]:
                x = F(i-1, j) + F(i, j - A[i])
            
            memo[(i,j)] = x
            return x
        return F(len(A)-1, amount)

# 零钱组合/与顺序无关的组合

class Solution(object):
    def change(self, amount, coins):
        
        dp = [[0] * (1 + amount) for _ in range(1 + len(coins))]
        dp[0][0] = 1
        
        for i in range(1, 1 + amount):
            if i % coins[0] == 0:
                dp[i] = 1
        
        for i in range(2, 1 + len(coins)):
            for j in range(0, 1 + amount):
                if j < coins[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j - coins[i-1]]
        return dp[-1][-1]

                
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# 零钱最小数量

class Solution(object):
    def coinChange(self, coins, amount):
        
        dp = [1 + amount] * (1 + amount)
        dp[0] = 0
        
        for i in range(1, 1 + amount):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], 1 + dp[i - c])

        if dp[-1] == 1 + amount:
            return -1
        else:
            return dp[-1]


# 377. 组合总和 Ⅳ 
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        
        dp = [0] * (1 + target)
        dp[0] = 1
        
        for i in range(1, len(dp)):
            for choice in nums:
                if i >= choice:
                    dp[i] += dp[i - choice]
        return dp[-1]
    

class Solution(object):
    def findTargetSumWays(self, nums, S):
        """
        :type nums: List[int]
        :type S: int
        :rtype: int
        """
        
        
        self.r = 0
        def F(path, start):
            if len(path) == len(nums):
                if sum(path) == S:
                    self.r += 1
                return
            F(path + [nums[start]], 1 + start)
            F(path + [-nums[start]], 1 + start)
        F([], 0)
        return self.r


# 是否可以分割 0-1背包问题
class Solution(object):
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        
        s = sum(nums) % 2
        if s == 1:
            return False
        
        dp = [[False] * (1 + s) for _ in range(1 + len(nums))]
        
        for i in range(1 + len(nums)):
            dp[i][0] = True
        
        for i in range(1, 1 + len(nums)):
            for j in range(1, 1 + s):
                if j < nums[i-1]:
                    dp[i][j] = dp[i-1][j]
                else:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]
        return dp[-1][-1]
          
                
## 单词分割(错误解法)
# class Solution:
#     def wordBreak(self, s, wordDict): 
#         dp = [[False] * (1 + len(s)) for _ in range(1 + len(wordDict))]
        
#         for i in range(1 + len(wordDict)):
#             dp[i][0] = True
        
#         for i in range(1, 1 + len(wordDict)):
#             for j in range(1, 1 + len(s)):
#                 if j >= len(wordDict[i-1]) and wordDict[i-1] == s[j-len(wordDict[i-1]) : j]:
#                     dp[i][j] = dp[i-1][j] or dp[i][j - len(wordDict[i-1])]
#                 else:
#                     dp[i][j] = dp[i-1][j]
#         print dp
#         return dp[-1][-1]

# 单词切割 完全背包问题(考虑顺序)
class Solution:
    def wordBreak(self, s, wordDict): 
        dp = [True] + [False] * len(s)
        
        for i in range(1, len(dp)):
            for w in wordDict:
                if i >= len(w) and s[i-len(w) : i] == w:
                    dp[i] = dp[i] or dp[i - len(w)]
        return dp[-1]
 

# 数组是否可以分割为两部分使得两部分的和相等 0-1背包问题 
class Solution(object):
    def canPartition(self, nums):
        s = sum(nums)
        if s % 2 == 1:
            return False
        
        target = s / 2
        dp = [True] + [False] * target
        
        for num in nums:
            for j in range(len(dp), -1, -1):
                if j >= num:
                    dp[j] = dp[j] or dp[j - num] # (放 or 不放)
        return dp[-1]


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
            for k in range(K, 0, -1):
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


# 136 singlenum位运算 
class Solution(object):
    def singleNumbers(self, nums):
        xor = reduce(lambda x,y: x ^ y, nums)
        
        mask = 1
        while mask & xor == 0:
            mask <<= 1 # 假设两个singlenumber是a,b，mask就是第一个a,b不相等的低位，通过mask对a, b分组
        
        a, b = 0, 0
        for n in nums:
            if mask & n == 0:
                a ^= n
            else:
                b ^= n
        return [a, b]

# 714 
class Solution(object):
    def maxProfit(self, prices, fee):
        dp_0, dp_1 = 0, -prices[0] - fee
        
        for i in range(1, len(prices)):
            dp_0 = max(dp_0, dp_1 + prices[i])
            dp_1 = max(dp_1, dp_0 - fee - prices[i])
        return dp_0


class Solution(object):
    def minSteps(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        def F(n):
            if n == 1:
                return 0
            for i in range(int(n/2), 1, -1):
                if n % i == 0:
                    return F(i) + n / i
            return n
        return F(n)
            

# 583 编辑距离            
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        
        import numpy as np
        
        s = word1
        t = word2
        
        M = len(s)
        N = len(t)
        
        dp = np.zeros((M+1, N+1)).tolist()
        
        for j in range(N+1):
            dp[0][j] = j
        for i in range(M+1):
            dp[i][0] = i
        
        for i in range(1, 1+M):
            for j in range(1, 1+N):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        1 + dp[i-1][j],
                        1 + dp[i][j-1],
                        1 + dp[i-1][j-1]
                    )
        return dp[-1][-1]


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
        
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        self.r = -float('inf')
        
        def F(root):
            if not root:
                return 0
            
            left = max(0, F(root.left))
            right = max(0, F(root.right))
            self.r = max(self.r, left + right + root.val)
            return max(left, right) + root.val

        F(root)
        return self.r



# 124. 通过前序遍历和中序遍历的值恢复二叉树
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        
        if not inorder:
            return None
        
        root = TreeNode(preorder[0])
        
        mid = inorder.index(preorder[0])
        
        root.left = self.buildTree(preorder[:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root


# 99 修复错误的BST
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution(object):
    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        
        self.pre = TreeNode(float('-inf'))
        self.first = None
        self.second = None
        
        def F(root):
            
            if not root:
                return 
            
            F(root.left)
            
            if not self.first and root.val < self.pre.val:
                self.first = self.pre
                self.second = root
            elif self.first and root.val < self.pre.val:
                self.second = root
                
            self.pre = root
            
            F(root.right)
        
        
        F(root)
        self.first.val, self.second.val = self.second.val, self.first.val

        
# 160. 相交链表
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        
        pa, pb = headA, headB
        
        while pa != pb:
            pa = pa.next if pa else headB
            pb = pb.next if pb else headA
        return pa

# 206. 反转链表
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

# class Solution(object):
#     def reverseList(self, head):
#         """
#         :type head: ListNode
#         :rtype: ListNode
#         """
        
#         pre = None
#         cur = head
#         while cur:
#             t = cur.next
#             cur.next = pre
#             pre = cur
#             cur = t
#         return pre


class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if not head:
            return None
        
        # 递归出口
        if not head.next:
            return head
        
        # 拆解为子问题
        last = self.reverseList(head.next)
        
        # 所有子问题的相同逻辑
        head.next.next = head
        head.next = None
        
        return last


# 反转链表的前 n 个节点
class Solution(object):
    def reverseN(self, head, n):
        
        if not head:
            return None
        
        if n == 1:
            con = head.next
            return head
        
        last = self.reverseN(head.next, n - 1)
        
        head.next.next = head
        head.next = con
        
        return last


# 92. 反转链表 II
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    
    con = None
    
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        
        def reverseTopN(head, n):
            if not head:
                return None
            if n == 1:
                self.con = head.next
                return head
            
            last = reverseTopN(head.next, n - 1)
            head.next.next = head
            head.next = self.con
            return last
        
        if m == 1:
            return reverseTopN(head, n)
        
        head.next = self.reverseBetween(head.next, m-1, n-1)
        
        return head
    

# 反转a、b之间的节点,其中a是头结点
class Solution:
    con = None
    def reverseBetweenAB(self, a, b):
        if a == b:
            self.con = a.next
            return a
        last = self.reverseBetweenAB(a.next, b)
        a.next.next = a
        a.next = self.con
        return last
        

# 25. K 个一组翻转链表   
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    
    con = None
    
    def reverseBetween(self, a, b):
        
        # 迭代法
        # pre = None
        # cur = a
        # while cur != b:
        #     t = cur.next
        #     cur.next = pre
        #     pre = cur
        #     cur = t
        # return pre
        
        # 递归法
        if a == b:
            self.con = a.next
            return a
        last = self.reverseBetween(a.next, b)
        a.next.next = a
        a.next = self.con
        return last
    
    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """

        if not head: 
            return None
        
        a = b = head
        
        for _ in range(k):
            if not b:
                return head
            b = b.next
        
        new_head = self.reverseBetween(a, b)
        a.next = self.reverseKGroup(b, k)
        
        return new_head  
    
    
# 21. 合并两个有序链表   
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        a, b = l1, l2
        
        if not a: 
            return b
        if not b:
            return a
        
        if a.val <= b.val:
            a.next = self.mergeTwoLists(a.next, b)
            return a
        else:
            b.next = self.mergeTwoLists(a, b.next)
            return b
        
    
    
# 83. 删除排序链表中的重复元素    
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """  
        
        if not head or not head.next:
            return head
        
        head.next = self.deleteDuplicates(head.next)
        
        if head.val == head.next.val:
            return head.next
        else:
            return head

                
    
    
    
    



        
        
        