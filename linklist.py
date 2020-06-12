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
        if a.next == b:
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
        
        # 第二步才轮到考虑子问题
        head.next = self.deleteDuplicates(head.next)
        
        # 第一步应该先考虑 return 的结果，这里要分情况讨论
        if head.val == head.next.val:
            return head.next
        else:
            return head
        

# 19. 删除链表的倒数第N个节点
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    
    """
    为何可以用递归方法？因为递归分为递和归两个过程，递的过程为从head开始依次询问next节点是倒序的第几个节点，这样一路询问到链表最后一个节点的next时结 束，此时为null，且为倒序的第0个节点，这样在归的过程中可以依次得到每个节点的倒序位置，当倒序位置与待删除的倒数位置相等时，即找到了待删除节点，此时 返回的为待删除节点的next节点，这样就可以在返回链表时跨过待删除节点，达到删除节点的目的。其他时候直接返回当前节点即可。
    
    https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/solution/di-gui-qi-shi-hen-rong-yi-by-antandcoffee/
    这个题解的【评论】很好
    """
    
    count = 0
    
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        
        if not head:
            return None
        
        head.next = self.removeNthFromEnd(head.next, n)
        self.count += 1
        
        if n == self.count:
            return head.next
        else:
            return head


# 24. 两两交换链表中的节点
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        
        if not head or not head.next:
            return head
        
        a, b = head, head.next
        a.next = self.swapPairs(head.next.next)
        b.next = a
        
        return b
    

# 445. 两数相加 II  
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    
    count = 0
    sum = 0
    
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        def F(head):
            if not head:
                return 
            F(head.next)
            self.count += 1
            self.sum += head.val * (10 ** (self.count - 1))
        
        self.count, self.sum = 0, 0
        F(l1)
        s1 = self.sum
        
        self.count, self.sum = 0, 0
        F(l2)
        s2 = self.sum
        
        s = str(s1 + s2)
        
        head = p = ListNode(int(s[0]))
        for i in range(1, len(s)):
            p.next = ListNode(int(s[i]))
            p = p.next
        return head
    

        
# 回文联表        
class Solution(object):
    def isPalindrome(self, head):
        fast = slow = head
        while fast and fast.next:
            fast =  fast.next.next
            slow = slow.next
        
        def reverse(a, b): 
            if a == b:
                return a
            
            if a.next == b:
                return a
            last = reverse(a.next, b)
            a.next.next = a
            a.next = b
            return last
        
        p = reverse(head, slow)
        
        if fast:
            slow = slow.next
        
        while slow:
            if p.val != slow.val:
                return False
        return True


# 328. 奇偶链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def oddEvenList(self, head):
        if not head:
            return head
        
        odd = head
        even = even_head = head.next
        
        while odd.next and even.next:
            odd.next = odd.next.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        odd.next = even_head
        return head


class Solution(object):
    def splitListToParts(self, root, k):
        n = 0
        p = root
        while p:
            p = p.next
            n += 1
        parts, reminder = divmod(n, k)
        
        ret = []
        cur = root
        for i in range(k):
            head = p = ListNode(None)
            for j in range(parts + (i<reminder)):
                p.next = ListNode(cur.val)
                p = p.next
                if cur:
                    cur = cur.next
            ret.append(head.next)
        return ret
