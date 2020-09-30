class Solution(object):
    def partitionLabels(self, S):

        last = {c:i for i,c in enumerate(S)}
        
        start = 0; end = last[S[0]]; cnt = 0; r = []
        for i,c in enumerate(S):
            end = max(end, last[c])
            if i == end:
                r.append(end-start+1)
                if i < len(S)-1:
                    start = i+1
                    end = last[S[i+1]]
        return r


class Solution(object):
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        dp = [1] * len(nums)
        for i in range(1, len(nums)):
            for j in range(i+1):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], 1+dp[j])
        return max(dp)
    
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        
        dp = [1 + amount] * (1 + amount)
        dp[0] = 0
        for i in range(1, 1+amount):
            for c in coins:
                if i >= c:
                    dp[i] = min(dp[i], dp[i-c])
        if dp[-1] == 1 + amount: return -1
        return dp[-1]


class Solution(object):
    def change(self, amount, coins):
        """
        :type amount: int
        :type coins: List[int]
        :rtype: int
        """
        if not coins: reutrn 0
        for coin in coins:
            for i in range(coin, 1+amount):
                dp[i] += dp[i-coin]
        return dp[-1]


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        
        n = len(s)
        dp = [False] * (1 + n)
        dp[0] = True
        for i in range(1, 1 + n):
            for word in wordDict:
                if len(word) <= i and s[i - len(word): i]:
                    dp[i] = dp[i] or dp[i - len(word)]
        return dp[-1]


class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        
        import numpy as np
        dp = np.zeros((len(prices, 2))).tolist()
        dp[0][0] = 0; dp[0][1] = -prices[0]
        for i in range(0, len(prices)):
            for j in range(2):
                dp[i][0] = max(dp[i-1][0], dp[i-1][1] + prices[i])
                dp[i][1] = max(dp[i-1][1], dp[i-1][0] - prices[i])
        return dp[-1][0]



class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        dp_pre_0, dp_i_0, dp_i_1 = 0, -prices[0] # -prices[0]第一天持有，不可能
        for i in range(1, len(prices)):
            tmp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, dp_pe_0 - prices[i])
            dp_pre_0 = tmp
        return dp_i_0
     
        
class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        dp_i_0, dp_i_1 = 0, -float('inf')
        for i in range(len(prices)):
            tmp = dp_i_0
            dp_i_0 = max(dp_i_0, dp_i_1 + prices[i])
            dp_i_1 = max(dp_i_1, tmp - prices[i] - fee)
        return dp_i_0


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        # self.ma = self.cnt = 0
        # def countOnes(n):
        #     if nums[n] == 0:
        #         self.ma = max(self.ma, self.cnt)
        #         self.cnt = 0
        #     if nums[n] == nums[n-1]:
        #         self.cnt += 1
        
        cur = ma = 0
        for n nums:
            cur = 0 if n==0 else 1+cur
            ma = max(ma, cur)
        return ma
                
        

class Solution(object):
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        if not matrix or not matrix[0]: return False
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n-1
        while row < m and col >=0:
            if matrix[row][col] == target: return True
            elif matrix[row][col] < target: row += 1
            else: col -= 1
        return False
    
    
class Solution(object):
    def arrayNesting(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        A = nums
        ma = 0
        for i in range(len(nums)):
            s = set()
            new_index = A[i]
            while new_index not in s:
                s.add(new_index)
                new_index = A[new_index]
            ma = max(ma, len(s))
        return ma


class Solution(object):
    def longestOnes(self, A, K):
        """
        :type A: List[int]
        :type K: int
        :rtype: int
        """
        
        from collections import Counter
        
        ma = i = j = 0
        lookup = Counter()
        while j < len(A):
            lookup[A[j]] += 1
            j += 1
            while lookup[0] > K:
                if A[i] == 0: 
                    lookup[0] -= 1
                i += 1
            if lookup[0] <= k:
                ma = max(ma, j-i)
        return ma        
    

class Solution(object):
    def findShortestSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        from collections import Counter
        
        d = Counter(nums)
        du = max(d.values())
        
        i = j = 0
        mi = len(nums)
        lookup = Counter()
        while j < len(nums):
            lookup[nums[j]] += 1
            j += 1
            while any(map(lambda x : x == du, lookup.values())):
                mi = min(mi, j - i)
                lookup[nums[i]] -= 1
                i += 1
        return mi


class Solution(object):
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        
        ss = set()
        for n in nums:
            if n-1 not in ss:
                cur_num = n
                cur_len = 1
                while cur_num not in ss:
                    cur_num += 1
                    cur_len += 1
            ma = max(ma, cur_len)
        return ma
                    
                
            
class Solution(object):
    def isPalindrome(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        def reverse(a, b):
            if a.next == b:
                return a
            last = reverse(a.next, b)
            a.next.next = a
            a.next = b
            return last
        
        if not head or not head.next: p = slow else p = reverse(head, slow)
        if fast: slow = slow.next
        
        while slow:
            if p.val != slow.val: return False
            slow = slow.next; fast = fast.next
        return True


class Solution(object):
    def splitListToParts(self, root, k):
        arr = []
        p = root
        while p:
            arr.append(p.val)
            p = p.next
        
        parts, mod = divmod(len(arr), k)
        ans = []
        cur = root
        for i in range(k):
            head = p = ListNode(None)
            for j in range(parts + (i < mod)):
                p.next = ListNode(cur.val)
                p = p.next
            if cur: cur = cur.next
            ans.append(head.next)
        
        return ans


class Solution:
    def oddEvenList(self, head):
        if not head: return head
        odd, even, even_head = head, head.next, head.next
        while even and even.next:
            odd.next = odd.next.next
            odd = odd.next
            even.next = even.next.next
            even = even.next
        odd = even_head
        return head
                

        
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        self.ans = True
        
        def maxDepth(root):
            l = maxDepth(root.left)
            r = maxDepth(root.right)
            if abs(l - r) > 1: self.ans = False
            return 1 + max(l, r)
        
        maxDepth(root)
        return self.ans
    
class Solution(object):   
    s = 0 
    def sumOfLeftLeaves(self, root):
        if not root: return 0
        if root.left and not root.left.left and root.left.right: self.root.left.val
        F(root.left)
        F(root.right)
    F(root)

    return self.s


class Solution(object):
    max_path = 0
    def longestUnivaluePath(self, root):
        def F(root):
            if not root: return 0
            l = F(root.left)
            r = F(root.right)
            lpath = 1 + l if root.left and root.val == root.val else 0
            rpath = 1 + r if root.right and root.val == root.val else 0
            max_path = max(max_path, lpath + rpath)
            return max(lpath, rpath)



class Solution(object):
    def findSecondMinimumValue(self, root):
        def F(root):
            if not root: return 
            
            if root.val < self.small:
                self.big = self.small
                self.small = root.val
            elif root.val < self.big and root.val > self.small:
                self.big = root.val
            
            F(root.left)
            F(root.right)
            
            
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
                
        ans = []
        cur, next = [root], []
        while cur:
            next = []
            ans.append(sum( x.val for x in cur ) * 1.0 / len(cur))
            for node in cur:
                if node.left: next.append(node.left)
                if node.right: next.append(node.right)
            cur = next
        return ans
            
        
class Solution(object):
    def findBottomLeftValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        cur = [root]
        while cur:
            next = []; ans = cur[0].val
            for node in cur:
                if node.left: next.append(node.left.val)
                if node.right: next.append(node.right.val)
            cur = next
        return ans
    
    
class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        
        if not root: return None
        
        if root.val > R: return self.trimBST(root.left, L, R)
        if root.val < L: return self.trimBST(root.right, L, R)
        root.left = self.trimBST(root.left, L, R)
        root.right = self.trimBST(root.right, L, R)
        
        return root
        

# 236. 二叉树的最近公共祖先(不好理解)
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        if not root or root.val == p.val or root.val == q.val: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if not left and not right: return None
        if not left: return right
        if not right: return left
        return root
    

class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        
        def findMid(head, tail):
            slow, fast = head, head.next
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
            return slow

        def F(head, tail):
            if head == tail: return
            mid = findMid(head, tail)
            root = TreeNode(mid.val)
            root.left = F(head, mid)
            root.right = F(mid.next, tail)
            return root

        return F(head, None)
    
    
class Solution(object):
    def dailyTemperatures(self, T):
        """
        :type T: List[int]
        :rtype: List[int]
        """
        
        ans = [0]*len(T)
        s = []
        for i in range(len(T)-1, -1, -1):
            while s and T[i] >= s[-1][-1]: s.pop()
            ans[i] = s[-1][0] - i if s else 0
            s.append((i, T[i]))
        return ans
 
    
class Solution(object):
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """

        s = nums[:-1][::-1]; ans = [0]*len(nums)
        for i in range(len(nums)-1, -1, -1):
            while s and nums[i] >= s[-1]: s.pop()
            ans[i] = s[-1] if s else -1
            s.append(nums[i])
        return ans
            
        
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        
        i,j = 0, len(nums)-1
        while i < j:
            m = (i+j) >> 1
            if nums[m] > nums[j]: i = m + 1
            else: j = m
        splitPoint = i
        
        def find(A, key):
            i,j = 0, len(A)-1
            while i < j:
                m = (i + j) >> 1
                if A[m]==key: return m
                elif A[m] < key: i = m+1
                else: j = m-1
            return -1

        idx1, idx2 = find(nums[:splitPoint], target), find(nums[splitPoint:], target)
        if idx1==idx2==-1: return -1
        elif idx1==-1: return idx2
        elif idx2==-1: return idx1
        
        
class Solution(object):
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        
        s = sum(abs(arr[i]-x) for i in range(k))
        best = s
        
        lo, hi = 0, k-1
        for i in range(k, len(arr)):
            s += abs(arr[i]-x) - abs(arr[i-k]-x)
            if s < best:
                best = s
                lo += 1
        return arr[lo:lo+k]



def find_pairs(arr, k):
    arr.sort()
    i, j = 0, len(arr)-1
    ans = None
    min_val = float('inf')
    while i < j:
        val = arr[i] + arr[j]
        if val < min_val:
            ans = [arr[i], arr[j]]
            min_val = val
    return ans
        
        