

# 104. Maximum Depth of Binary Tree (Easy)
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):    
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


# 110. 平衡二叉树
class Solution(object):
    
    def maxDepth(self, root):
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
    
    def isBalanced(self, root):
        
        """
        :type root: TreeNode
        :rtype: bool
        """
        
        if not root:
            return True
        
        return abs(self.maxDepth(root.left) - self.maxDepth(root.right)) <= 1 and \
    self.isBalanced(root.left) and self.isBalanced(root.right)
    
    
# 543. 二叉树的直径
class Solution(object):  
    maxDepth = 0  
    def diameterOfBinaryTree(self, root):
        def F(root):
            if not root:
                return 0
            left = F(root.left)
            right = F(root.right)
            self.maxDepth = max(self.maxDepth, left + right)
            return 1 + max(left, right)

# 翻转二叉树         
class Solution(object):
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        
        def F(root):
            if not root:
                return None
            t = root.left
            root.left = F(root.right)
            root.right = F(t)
            return root
        return F(root)

# 合并二叉树
class Solution(object):
        
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """      
        
        if not t1 and not t2:
            return None
        if t1 and not t2:
            return TreeNode(t1.val)
        if not t1 and t2:
            return TreeNode(t2.val)
        
        root = TreeNode(t1.val + t2.val)
        root.left = self.mergeTrees(t1.left, t2.left)
        root.left = self.mergeTrees(t1.right, t2.right)
        return root
  
# 112. 路径总和 
class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        
        def F(root, s):
            if not root:
                return False
            if not root.left and not root.right and root.val == s:
                return True
            return F(root.left, s - root.val) or F(root.right, s - root.val)
        return F(root, sum)


# 437. 路径总和 III
class Solution(object): 
    count = 0   
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        
        def F(root, s):
            if not root:
                return
            if root.val == s:
                self.count += 1
            F(root.left, s - root.val) or F(root.right, s - root.val)
        
        def traverse(root):
            if not root:
                return 
            F(root, sum)
            traverse(root.left)
            traverse(root.right)
        
        traverse(root)
        return self.count


# 572. 另一个树的子树
#https://leetcode-cn.com/problems/subtree-of-another-tree/solution/dui-cheng-mei-pan-duan-zi-shu-vs-pan-duan-xiang-de/
class Solution(object):    
    def isSameTree(self, s, t):
        if not s and not t: return True
        if not s or not t: return False
        return s.val == t.val and self.isSameTree(s.left, t.left) and self.isSameTree(s.right, t.right)
    
    def isSubtree(self, s, t):
        if not s and not t: return True
        if not s or not t: return False
        return self.isSameTree(s, t) or self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
        
        
        
# 101. 对称二叉树  
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if not root: return True
        return self.mirror(root.left, root.right)

    def mirror(self, s, t):
        if not s and not t: return True
        if not s or not t: return False
        return s.val == t.val and self.mirror(s.left, t.right) and self.mirror(s.right, t.left)


# 111. 二叉树的最小深度
class Solution(object):
    ans = float('inf')
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if not root: return 0
        left = self.minDepth(root.left)
        right = self.minDepth(root.right)
        if not root.left: return 1 + right # root 只有一个子节点
        if not root.right: return 1 + left
        return 1 + min(left, right)
 
    
# 404. 左叶子之和
class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        def F(root):
            if not root: return
            if root.left and not root.left.left and not root.left.right: self.ans += root.left.val
            F(root.left)
            F(root.right)
        F(root)
        return self.ans


# 两个二叉树是否完全相同
def isSame(s, t):
    if not s and not t:
        return True
    if not s or not t:
        return False
    return isSame(s.left, t.left) and isSame(s.right, t.right)
        

# BST中是否包含target
def f(root, target):
    if not root:
        return False
    if root.val == target:
        return True
    # return f(root.left, target) or f(root.right, target)
    if root.val < target:
        return f(root.right, target)
    if root.val > target:
        return f(root.left, target)


# 98. 验证二叉搜索树
class Solution(object):
    pre = float('-inf')
    def isValidBST(self, root):
        if not root:
            return True
        if not self.isValidBST(root.left):
            return False
        
        if root.val <= self.pre:
            return False
        self.pre = root.val
        
        return self.isValidBST(root.right)
    

#  687. 最长同值路径       
class Solution(object):
    max_path = float('-inf')
    def longestUnivaluePath(self, root):
        def F(root):
            if not root:
                return 0
            left = F(root.left)
            right = F(root.right)
            left_path = 1 + left if root.left and root.left.val == root.val else 0
            right_path = 1 + right if root.right and root.right.val == root.val else 0
            self.max_path = max(self.max_path, left_path + right_path)
            return max(left_path, right_path)
        F(root)
        return self.max_path
    
    
# 337. 打家劫舍 III (没加备忘录)
class Solution(object):    
    def rob(self, root):
        if not root: 
            return 0
        
        do = root.val
        if root.left:
            do += self.rob(root.left.left) + self.rob(root.left.right)
        if root.right:
            do += self.rob(root.right.left) + self.rob(root.right.right)
        
        not_do = self.rob(root.left) + self.rob(root.right)
        
        return max(do, not_do)


# 671. 二叉树中第二小的节点
class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root: return -1
        if not root.left and not root.right: return -1
        left, right = root.left.val, root.right.val
        if root.left and root.val == root.left.val: left = self.findSecondMinimumValue(root.left)
        if root.right and root.val == root.right.val: right = self.findSecondMinimumValue(root.right)
        if left == -1: return right
        if right == -1: return left
        return min(left, right)
        

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


# 669. 修剪二叉搜索树
class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        
        def F(root): 
            if not root: return None
            elif root.val < L: return F(root.right)
            elif root.val > R: return F(root.left)
            else:
                root.left = F(root.left)
                root.right = F(root.right)
                return root
        
        return F(root)
    
# 230. 二叉搜索树中第K小的元素
class Solution(object):
    idx = 0
    r = -1
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        
        def F(root):
            if not root:
                return 
            F(root.left)
            self.idx += 1
            if self.idx == k:
                self.r = root.val
            F(root.right)
        
        
        F(root)
        return self.r


# 236. 二叉树的最近公共祖先(不好理解)
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        
        if not root or root == p or root == q: return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if not left and not right: return None
        if not left: return right
        if not right: return left
        return root

# 剑指 Offer 68 - I. 二叉搜索树的最近公共祖先
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        if root == p or root == q: return root
        if root.val < min(p.val, q.val): return self.lowestCommonAncestor(root.right, p, q)
        elif root.val > max(p.val, q.val): return self.lowestCommonAncestor(root.left, p, q)
        else: return root

# 剑指 Offer 07. 重建二叉树
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        if not preorder: return None
        mid = inorder.index(preorder[0])
        root = TreeNode(inorder[mid])
        root.left = self.buildTree(preorder[1:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root

# 108. 将有序数组转换为二叉搜索树   
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        
        i, j = 0, len(nums)-1
        if i > j: return None
        m = (i + j) >> 1
        root = TreeNode(nums[m])
        root.left = self.sortedArrayToBST(nums[:m])
        root.right = self.sortedArrayToBST(nums[m+1:])
        return root


# 109. 有序链表转换二叉搜索树
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        
        if not head: return None
        if not head.next: return TreeNode(head.val)
        pre = self.preMid(head)
        mid = pre.next
        pre.next = None
        root = TreeNode(mid.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(mid.next)
        return root
    
    def preMid(self, head):
        pre, slow, fast = head, head, head.next # pre 中点的前一个节点
        while fast and fast.next:
            pre = slow 
            slow = slow.next
            fast = fast.next.next
        return pre

# 两数之和 IV - 输入 BST      
class Solution(object):
    res = False
    s = set()
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        def F(root):
            if not root: return
            F(root.left)
            if k - root.val in self.r:
                self.r = True
            else:
                self.s.add(root.val)
            F(root.right)
        F(root)
        return self.r

# 530. 二叉搜索树的最小绝对差
class Solution(object):
    
    pre = -float('inf')
    min_val = float('inf')
    
    def getMinimumDifference(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        def F(root):
            if not root: return 
            F(root.left)
            min_val = min(abs(root.val - self.pre), min_val)
            self.pre = root.val
            F(root.right)
        F(root)
        return self.min_val


# 501. 二叉搜索树中的众数
class Solution:
    maxcount = 0
    count = 0
    pre = None
    r = []
    def findMode(self, root):
        def F(root):
            if not root: return 
            F(root.left)
            
            if self.pre == root.val: self.count += 1
            else: self.count = 1
            if self.count == self.maxcount:
                self.r.append(root.val)
            elif self.count > self.maxcount:
                self.maxcount = self.count
                self.r = [root.val]                
            
            self.pre = root.val            
            F(root.right)
        F(root)
        return self.r


# 124. 二叉树中的最大路径和
class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        self.r = -float('inf')
        
        def F(root):
            if not root: return 0
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
        if not inorder: return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[:mid+1], inorder[:mid])
        root.right = self.buildTree(preorder[mid+1:], inorder[mid+1:])
        return root
    
# 99 修复错误的BST
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
            
            if not root: return 
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
        
        
# 208. 实现 Trie (前缀树)
class Node(object):
    def __init__(self):
        self.word = None
        self.isEnd = False
        self.next = {}
        
class Trie(object):
    
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()


    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        p = self.root
        for char in word:
            if char not in p.next:
                p.next[char] = Node()
            p = p.next[char]
        p.isEnd = True
        p.word = word

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        p = self.root
        for char in word:
            if char in p.next:
                p = p.next[char]
            else:
                return False
        return p.isEnd
        
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        
        p = self.root
        for char in prefix:
            if char not in p.next:
                return False
            p = p.next[char]
        else:
            return True

# 677. 键值映射
class Node(object):
    def __init__(self):
        self.next = {}
        self.val = 0
        self.isEnd = False
        
class MapSum(object):
    sum = 0
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = Node()

    def insert(self, key, val):
        """
        :type key: str
        :type val: int
        :rtype: None
        """
        p = self.root
        for char in key:
            if char not in p.next:
                p.next[char] = Node()
            p = p.next[char]
        p.isEnd = True
        p.val = val

    def sum(self, prefix):
        """
        :type prefix: str
        :rtype: int
        """
        p = self.root
        for char in prefix[:-1]:
            if char not in p.next:
                return 0
            else:
                p = p.next[char]
        if p.val != prefix[-1]: return 0
        
        root = p
        self.s = 0
        def F(root):
            if root.isEnd: return 0
            for node in root.next:
                if node:
                    self.s += node.val
                F(node)
        F(root)
        return self.s
        
        
# 239. 滑动窗口最大值   

from collections import deque

class Q:
    def __init__(self, ksize):
        self.x = deque(maxlen=ksize)  
    
    def push(self, n):
        while self.x and n > self.x[-1]:
            self.x.pop()
        self.x.append(n)      
    
    def max(self):
        if self.x: return self.x[0]
    
    def popleft(self, n):
        if self.x and n == self.x[0]: 
            self.x.popleft()
        
        
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        r = []
        q = Q(ksize=k)
        for i, n in enumerate(nums):
            q.push(n)
            if i >= k - 1:
                r.append(q.max())
                q.popleft(nums[i-k+1])
        return r
           
        
class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        
        if len(s) != len(t): return False
        c = {}
        d = {}
        for (x,y) in zip(s,t):
            if x not in d:
                d[x] = y
            if y not in c:
                c[y] = x
            if x in d and d[x] != y: return False
            if y in c and c[y] != x: return False   
        return True
            