

        
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
    
# 141. 环形链表 
class Solution:
    def hasCycle(self , head ):
        # write code here
        
        if not head or not head.next: return False
        slow = fast = head
        while 1:
            if not (fast and fast.next): return False
            slow, fast = slow.next, fast.next.next
            if slow == fast: return True
            

# 142. 环形链表 II (返回链表开始入环的第一个节点) 
# https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/linked-list-cycle-ii-kuai-man-zhi-zhen-shuang-zhi-/
class Solution(object):
    def detectCycle(self, head):
        fast, slow = head, head
        while True:
            if not (fast and fast.next): return
            fast, slow = fast.next.next, slow.next
            if fast == slow: break
        fast = head
        while fast != slow:
            fast, slow = fast.next, slow.next
        return fast

# 206. 反转链表
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
        # 递归出口
        if not head or not head.next: return None
        # 拆解为子问题
        last = self.reverseList(head.next)
        # 所有子问题的相同逻辑
        head.next.next = head
        head.next = None
        return last


# 反转链表的前 n 个节点
class Solution(object):
    def reverseN(self, head, n):
        if not head or not head.next: return None
        if n == 1:
            con = head.next
            return head
        last = self.reverseN(head.next, n - 1) # 问题规模减小的方向
        head.next.next = head
        head.next = con
        return last


# 92. 反转链表 II
class Solution(object):
    con = None
    def reverseTopN(self, head, n):
            if not head: return None
            if n == 1:
                self.con = head.next
                return head
            last = self.reverseTopN(head.next, n - 1)
            head.next.next = head
            head.next = self.con
            return last
    
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if m == 1: return self.reverseTopN(head, n)
        head.next = self.reverseBetween(head.next, m-1, n-1)
        return head
        

# 25. K 个一组翻转链表   
class Solution(object):
    con = None
    # 反转a、b之间的节点,其中a是头结点
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
            self.con = b
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
        if not head:  return None
        a = b = head
        for _ in range(k):
            if not b: return head
            b = b.next
        
        new_head = self.reverseBetween(a, b)
        a.next = self.reverseKGroup(b, k)
        
        return new_head  
    
    
# 21. 合并两个有序链表   
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        a, b = l1, l2
        if not a:  return b
        if not b: return a
        
        if a.val < b.val: 
            a.next = self.mergeTwoLists(a.next, b) # 指向下一个最小值
            return a # 谁的值小，谁就作为头儿返回
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
        if not head or not head.next: return head
        # 第二步才轮到考虑子问题
        head.next = self.deleteDuplicates(head.next)
        # 第一步应该先考虑 return 的结果，这里要分情况讨论
        if head.val == head.next.val: return head.next
        else: return head

# 82 删除排序链表中的重复元素 II
class Solution(object):
    def deleteDuplicates(self, head):
        if not head or not head.next: return head
        if head.next and head.val==head.next.val:
            while head.next and head.val==head.next.val:
                head = head.next
            return self.deleteDuplicates(head.next)
        else:
            head.next = self.deleteDuplicates(head.next)
            return head
        

# 19. 删除链表的倒数第N个节点
class Solution(object):
    
    """
    为何可以用递归方法？因为递归分为递和归两个过程，递的过程为从head开始依次询问next节点是倒序的第几个节点，这样一路询问到链表最后一个节点的next时结 束，此时为null，且为倒序的第0个节点，这样在归的过程中可以依次得到每个节点的倒序位置，当倒序位置与待删除的倒数位置相等时，即找到了待删除节点，此时 返回的为待删除节点的next节点，这样就可以在返回链表时跨过待删除节点，达到删除节点的目的。其他时候直接返回当前节点即可。
    
    https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/solution/di-gui-qi-shi-hen-rong-yi-by-antandcoffee/
    这个题解的【评论】很好
    
    为何可以用递归方法？因为递归分为递和归两个过程，递的过程为从head开始依次询问next节点是倒序的第几个节点，这样一路询问到链表最后一个节点的next时结 束，此时为null，且为倒序的第0个节点，这样在归的过程中可以依次得到每个节点的倒序位置，当倒序位置与待删除的倒数位置相等时，即找到了待删除节点，此时 返回的为待删除节点的next节点，这样就可以在返回链表时跨过待删除节点，达到删除节点的目的。其他时候直接返回当前节点即可。
    
    """
    
    count = 0
    
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        
        if not head: return None
        
        head.next = self.removeNthFromEnd(head.next, n)
        self.count += 1
        
        if n == self.count: return head.next # 此时的节点为待删除节点，返回其next节点跨过待删除节点，达到删除节点的目的
        else: return head # 其他情况下正常返回本节点


# 24. 两两交换链表中的节点
class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return head
        a, b = head, head.next
        a.next = self.swapPairs(b.next)
        b.next = a
        return b
    

# 445. 两数相加 II  
class Solution:
    s = ''
    def addTwoNumbers(self, l1, l2):
        def F(head):
            if not head: return
            F(head.next)
            self.s = str(head.val) + self.s 
        
        self.s = ''; F(l1); n1 = int(self.s)
        self.s = ''; F(l2); n2 = int(self.s)
        
        p = head = ListNode(-1)
        for char in str(n1 + n2):
            p.next = ListNode(int(char))
            p = p.next
        return head.next
    

        
# 回文联表        
class Solution(object):
    def isPalindrome(self, head):
        
        # 令fast指向末端；slow指向中点；
        fast = slow = head
        while fast and fast.next:
            fast =  fast.next.next
            slow = slow.next
        
        # 反转操作，以a为头结点
        def reverse(a, b): 
            if a == b: return a
            if a.next == b: return a
            last = reverse(a.next, b)
            a.next.next = a
            a.next = b
            return last
        
        # p是反转操作后新的头结点
        p = reverse(head, slow)
        
        # 若fast不是None，说明链表数字是奇数个
        if fast: slow = slow.next
        
        while slow:
            if p.val != slow.val: return False
            slow = slow.next; p = p.next
        return True


# 328. 奇偶链表
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head or not head.next: return head
        odd, even, evenhead = head, head.next, head.next
        while even and even.next:
            odd.next = odd.next.next
            even.next = even.next.next
            odd = odd.next
            even = even.next
        odd.next = evenhead
        return head

# 86. 分隔链表
# 和奇偶链表像
#https://leetcode-cn.com/problems/partition-list/solution/liang-ge-dummyran-hou-pin-jie-by-powcai/
class Solution:
    def partition(self , head , x ):
        # write code here
        p1 = head1 = ListNode(-1)
        p2 = head2 = ListNode(-1)
        while head:
            if head.val < x:
                p1.next = head
                p1 = p1.next
            else:
                p2.next = head
                p2 = p2.next
            head = head.next
        p1.next = head2.next
        p2.next = None
        return head1.next

# 143. 重排链表
# https://leetcode-cn.com/problems/reorder-list/solution/zhao-dao-zhong-dian-fan-zhuan-hou-xu-lian-biao-zho/
# 遍历直到，从头开始的链表，和从尾开始的链表都到达了终点。
class Solution:
    def reorderList(self, head):
        """
        Do not return anything, modify head in-place instead.
        """

        def middleNode(head):
            slow = fast = head
            while fast and fast.next: slow, fast = slow.next, fast.next.next
            return slow

        def reverse(head):
            if not head or not head.next: return head
            last = reverse(head.next)
            head.next.next = head
            head.next = None
            return last
        
        def merge(left, right):
            p1, p2 = left, right
            while p1 and p2:
                t1, t2 = p1.next, p2.next
                p1.next = p2
                p2.next = t1
                p1, p2 = t1, t2

        if not head:
            return None
        middle = middleNode(head) # 找中点
        l2 = middle.next # 第二个链表开始
        l1 = head # 第一个链表开始
        middle.next = None # 第一个链表结束
        l2 = reverse(l2) # 翻转第二个
        merge(l1,l2) # 错位合并


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
